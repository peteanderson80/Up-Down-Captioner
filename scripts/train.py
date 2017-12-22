#!/usr/bin/env python
"""
Trains a model using one or more GPUs.
"""
from multiprocessing import Process

import caffe


def train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        weights, # pretrained weights to use
        gpus,  # list of device ids
        timing=False,  # show timing info for compute and communications
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.init_log(0, 1)
    caffe.log('Using devices %s' % str(gpus))

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver, snapshot, weights, gpus, timing, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))


def track_memory(net):
    print 'Memory Usage:'
    total = 0.0
    data = 0.0
    params = 0.0
    for k,v in net.blobs.iteritems():
        gb = float(v.data.nbytes)/1024/1024/1024
        print '%s : %.3f GB %s' % (k,gb,v.data.shape)
        total += gb
        data += gb
    print 'Memory Usage: Data %.3f GB' % data
    for k,v in net.params.iteritems():
        for i,p in enumerate(v):
            gb = float(p.data.nbytes)/1024/1024/1024
            total += gb
            params += gb
            print '%s[%d] : %.3f GB %s' % (k,i,gb,p.data.shape)
    print 'Memory Usage: Params %.3f GB' % params
    print 'Memory Usage: Total %.3f GB' % total

    
def solve(proto, snapshot, weights, gpus, timing, uid, rank):

    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)
    elif weights and len(weights) != 0:
        print ('Loading pretrained model '
                   'weights from {:s}').format(weights)
        solver.net.copy_from(weights)
        
    # For RCNNDataLayer, split the dataset across gpus
    if solver.net.layers[0].type == "Python":
        solver.net.layers[0].load_dataset(rank, len(gpus))
            
    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    solver.step(solver.param.max_iter)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--solver", required=True, help="Solver proto definition.")
    parser.add_argument("--snapshot", help="Solver snapshot to restore.")
    parser.add_argument("--weights", help="Pretrained weights to initialize finetuning.")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0],
                        help="List of device ids.")
    parser.add_argument("--timing", action='store_true', help="Show timing info.")
    args = parser.parse_args()

    train(args.solver, args.snapshot, args.weights, args.gpus, args.timing)
