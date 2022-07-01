all: test_gpu

test_gpu : test_gpu.cu
	nvcc $< -o test_gpu

clean:
	rm test_gpu