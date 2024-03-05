# DDP mode
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = select_device(opt.device, batch_size=opt.batch_size)
else:
    device = torch.device("mps")
if LOCAL_RANK != -1:
    msg = 'is not compatible with YOLO Multi-GPU DDP training'
    assert not opt.image_weights, f'--image-weights {msg}'
    assert not opt.evolve, f'--evolve {msg}'
    assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
    assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
    assert torch.backends.mps.is_available(), "MPS not available. Check PyTorch build, macOS version, and hardware"
    # assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
    device = torch.device("mps")
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
