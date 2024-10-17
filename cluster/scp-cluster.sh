#! /bin/sh

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

HOST_PATH=/home/giovanni.santini/parallel_computing

if [ ! -f $HOST_PATH/$1 ]; then
    echo "File $1 not found in host directory: $HOST_PATH"
    exit 1
fi

CLUSTER_PATH=/home/giovanni.santini/parallel_computing
CLUSTER_USER=giovanni.santini
CLUSTER_HOST=hpc.unitn.it

scp -r $HOST_PATH/$1 $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_PATH/$1
