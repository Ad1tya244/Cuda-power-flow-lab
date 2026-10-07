# Makefile - CUDA power-grid load flow case study
#   make                 build ./pf_cuda (default GPU arch sm_61 = GeForce MX250 / Pascal)
#   make ARCH=sm_75      Tesla T4 (Google Colab); sm_86 = RTX 30xx; sm_89 = RTX 40xx
#   make run             demo + verification + benchmarks

NVCC     ?= nvcc
ARCH     ?= sm_61
CCBIN    ?=
NVFLAGS  := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
HOSTOPT  := -Xcompiler -O3,-march=native,-fopenmp
LIBS     := -lcusparse -lcublas -lgomp
ifneq ($(CCBIN),)
  NVFLAGS += -ccbin $(CCBIN)
endif

pf_cuda: src/pf_cuda.cu src/powerflow.h
	$(NVCC) $(NVFLAGS) $(HOSTOPT) src/pf_cuda.cu -o $@ $(LIBS) $(EXTRA_LDFLAGS)

run: pf_cuda
	./pf_cuda all

lab:
	@./run.sh

clean:
	rm -f pf_cuda

.PHONY: run clean lab
