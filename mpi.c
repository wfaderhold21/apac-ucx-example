#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ucp/api/ucp.h>
#include <mpi.h>

#include "comm-mpi.h"
#include "errors.h"
#include "common.h"

MPI_Datatype mpi_buffer_exchange_dt; 

int mpi_worker_exchange(void *** param_worker_addrs)
{
    void ** worker_addresses;
    size_t worker_len;
    void * worker_address;
    size_t * worker_sizes;
    int error;
    int i;
    void * buf, * w_addr;
    size_t max = 0;
    int ret = 0;

    /* allocate */
    worker_addresses = (void **) malloc(sizeof(void *) * size);
    if (NULL == worker_addresses) {
        return ERR_NO_MEMORY;
    }


    error = ucp_worker_get_address(ucp_worker,
                                   (ucp_address_t **) &worker_address,
                                   &worker_len);
    if (error < 0) {
        free(worker_addresses);
        return -1;
    }

    worker_sizes = (size_t *) malloc(sizeof(size_t) * size);
    worker_sizes[my_pe] = worker_len;

    /* exchange */
    error = MPI_Allgather(&worker_sizes[my_pe],
                          1,
                          MPI_UNSIGNED_LONG,
                          worker_sizes,
                          1,
                          MPI_UNSIGNED_LONG,
                          MPI_COMM_WORLD);
    if (error != MPI_SUCCESS) {
        ret = -1;
        goto fail_exchange;
    }

    for (int j = 0; j < size; j++) {
        if (max < worker_sizes[j]) {
            max = worker_sizes[j];
        }
    }

    /* set up */
    for (i = 0; i < size; i++) {
        worker_addresses[i] = malloc(worker_sizes[i]);
        if (NULL == worker_addresses[i]) {
            ret = ERR_NO_MEMORY;
            goto fail_setup;
        }
    }
    buf = malloc(max * size);

    w_addr = malloc(max);
    memcpy(w_addr, worker_address, worker_len);
    error = MPI_Allgather(w_addr,
                          max,
                          MPI_BYTE,
                          buf,
                          max,
                          MPI_BYTE,
                          MPI_COMM_WORLD);
    if (error != MPI_SUCCESS) {
        ret = -1;
        goto fail_setup;
    }

    for (int i = 0; i < size; i++) {
        memcpy(worker_addresses[i], buf + (max * i), worker_sizes[i]);
    }
    free(buf);

    *param_worker_addrs = worker_addresses;
    
    return ret;

fail_setup:
    for (--i; i >= 0; i--) {
        free(worker_addresses[i]);
    }
fail_exchange:
    free(worker_sizes);
    free(worker_addresses);
    free(endpoints);

    return ret;
}

int mpi_buffer_exchange(void * buffer,
                        void *** pack_param,
                        uint64_t * remotes,
                        void * register_buffer)
{
    int error = 0;
    void ** pack = NULL;
    struct data_exchange * dx;
    struct data_exchange * rx = NULL;
    ucp_mem_h * mem = (ucp_mem_h *)register_buffer;
    size_t pack_size; 
    int ret = 0, i;
    ucs_status_t status;
    void * buf;
    void * packed_rkey;
    size_t max = 0;

    pack = (void **) malloc(sizeof(void *) * size);
    if (NULL == pack) {
        ret = ERR_NO_MEMORY;
        goto fail_mpi;
    }

    status = ucp_rkey_pack(ucp_context, *mem, &pack[my_pe], 
                           &pack_size);
    if (status != UCS_OK) {
        ret = error;
        goto fail_mpi;
    }

    remotes[my_pe] = (uint64_t)buffer;

    /* step 1: create some storage */
    rx = (struct data_exchange *)malloc(
                                    sizeof(struct data_exchange) * size);
    dx = (struct data_exchange *)malloc(sizeof(struct data_exchange));
    dx->pack_size = pack_size;
    dx->remote = remotes[my_pe];

    /* step 2: perform the allgather on the data */
    MPI_Allgather(dx, 
                  1, 
                  mpi_buffer_exchange_dt, 
                  rx, 
                  1, 
                  mpi_buffer_exchange_dt, 
                  MPI_COMM_WORLD);

    max = pack_size;
    /* step 3: loop over rx and pull out the necessary parts */ 
    for (i = 0; i < size; i++) {
        if (i == my_pe) {
            continue;
        }
        /* store remote address */
        remotes[i] = rx[i].remote;
        if (max < rx[i].pack_size) {
            max = rx[i].pack_size;
        }
    }
    buf = malloc(max * size);
    packed_rkey = malloc(max);
    memcpy(packed_rkey, pack[my_pe], pack_size);

    MPI_Allgather(packed_rkey,
                  max,
                  MPI_BYTE,
                  buf,
                  max,
                  MPI_BYTE,
                  MPI_COMM_WORLD);
    for (i = 0; i < size; i++) {
        /* allocate space for packed rkeys */
        pack[i] = malloc(rx[i].pack_size);
        if (NULL == pack[i]) {
            ret = ERR_NO_MEMORY;
            goto fail_purge_arrays;
        }
        /* copy remote rkey into pack */
        memcpy(pack[i], buf + (max * i), pack_size);
    }
    free(buf);
    free(packed_rkey);
    free(rx);
    free(dx);

    *pack_param = pack; 

    return ret;

fail_purge_arrays:
    for (--i; i >= 0; i--) {
        free(pack[i]);
    }
fail_mpi:
    if (rx != NULL) {
        free(rx);
    }
    if (NULL != pack) {
        free(pack);
    }
    return ret;
}

void create_mpi_datatype(void)
{
    int buffer_nr_items = 2;
    MPI_Aint buffer_displacements[2];
    int buffer_block_lengths[2] = {1,1};
    MPI_Datatype buffer_exchange_types[2] = {MPI_UINT64_T,
                                             MPI_UINT64_T};

    buffer_displacements[0] = offsetof(struct data_exchange, pack_size);
    buffer_displacements[1] = offsetof(struct data_exchange, remote);

    /* create an exchange data type for group creation/buffer registration */
    MPI_Type_create_struct(buffer_nr_items, 
                           buffer_block_lengths, 
                           buffer_displacements, 
                           buffer_exchange_types, 
                           &mpi_buffer_exchange_dt);
    MPI_Type_commit(&mpi_buffer_exchange_dt);
}

int init_mpi(void)
{
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    create_mpi_datatype(); 
    return 0;
}

int finalize_mpi(void)
{
    MPI_Type_free(&mpi_buffer_exchange_dt);
    MPI_Finalize();
    return 0;
}
