
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -g -G -arch=sm_75

EXEC = main

SRCS = kernel.cu
OBJS = $(SRCS:.cu=.o)

$(EXEC): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

ifdef dbg
debug: $(EXEC)
	gdb -ex run --args ./$(EXEC) arg1 arg2 arg3
endif

clean:
	rm -f $(OBJS) $(EXEC)
