from ultralytics.utils.benchmarks import benchmark

# Benchmark on GPU
benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=320, int8=True)