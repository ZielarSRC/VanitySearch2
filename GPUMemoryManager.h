// GPUMemoryManager.h
#ifdef WITH_GPU
#pragma once
#include <CL/cl.h>
#include <vector>

class GPUMemoryManager {
    cl_context context;
    cl_device_id device;
    std::vector<cl_mem> allocations;
    
public:
    GPUMemoryManager(cl_context ctx, cl_device_id dev) 
        : context(ctx), device(dev) {}
    
    ~GPUMemoryManager() {
        release_all();
    }
    
    cl_mem allocate(size_t size, cl_mem_flags flags) {
        cl_int err;
        cl_mem buf = clCreateBuffer(context, flags, size, NULL, &err);
        if (err == CL_SUCCESS) {
            allocations.push_back(buf);
        }
        return buf;
    }
    
    void release_all() {
        for (auto buf : allocations) {
            clReleaseMemObject(buf);
        }
        allocations.clear();
    }
    
    void prefetch(cl_command_queue queue, cl_mem buf, size_t size) {
        clEnqueueMigrateMemObjects(queue, 1, &buf, 
            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, NULL);
    }
    
    void copy_async(cl_command_queue queue, 
                   cl_mem src, cl_mem dst, size_t size) {
        clEnqueueCopyBuffer(queue, src, dst, 0, 0, size, 
                           0, NULL, NULL);
    }
};
#endif