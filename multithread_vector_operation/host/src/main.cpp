// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

///////////////////////////////////////////////////////////////////////////////////
// This host program creates runs two instances of a class in two separate threads.
// Each instance uses a different kernel:
// First instance executes a vector addition kernel to perform:
//  C = A + B
// where A, B and C are vectors with N elements.
//
// Second instance executes a vector memberwise multiplication kernel to perform:
// C = A * B (memberwise)
// where A, B and C are vectors with N elements.

// This host program supports partitioning the problem across multiple OpenCL
// devices if available. If there are M available devices, the problem is
// divided so that each device operates on N/M points. The host program
// assumes that all devices are of the same type (that is, the same binary can
// be used), but the code can be generalized to support different device types
// easily.
//
// Verification is performed against the same computation on the host CPU.
///////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

//Thread library for each platform
#ifdef _WIN32 //Windows
#include <windows.h>
typedef HANDLE acl_thread_t;
#else         //Linux
#include <pthread.h>
typedef pthread_t acl_thread_t;
#endif

using namespace aocl_utils;
// OpenCL runtime configuration
cl_platform_id platform = NULL;
unsigned num_devices = 0;
scoped_array<cl_device_id> device; // num_devices elements
cl_context context = NULL;
cl_program program = NULL;

class Problem
{
public:
  unsigned N ; // Problem size
  const char* kernel_name;
  bool pass; // used for verification.
  Problem(unsigned _N,const char* _kernel_name):N(_N),kernel_name(_kernel_name){
  }

  scoped_array<cl_command_queue> queue; // num_devices elements
  scoped_array<cl_kernel> kernel; // num_devices elements
  scoped_array<cl_mem> input_a_buf; // num_devices elements
  scoped_array<cl_mem> input_b_buf; // num_devices elements
  scoped_array<cl_mem> output_buf; // num_devices elements

  // Problem data.
  scoped_array<scoped_aligned_ptr<float> > input_a, input_b; // num_devices elements
  scoped_array<scoped_aligned_ptr<float> > output; // num_devices elements
  scoped_array<scoped_array<float> > ref_output; // num_devices elements
  scoped_array<unsigned> n_per_device; // num_devices elements
  void init();
  void run();
  void verify();
  void cleanup();
  void start();
};

// Function prototypes
void* start_helper(void * arg);
int acl_thread_create(acl_thread_t *newthread, void *attr, void *(*start_routine) (void *), void *arg);
int acl_thread_join(acl_thread_t *threadp);
float rand_float();
bool init_opencl();
void cleanup_opencl();

// Entry point.
int main(int argc, char **argv) {
  Options options(argc, argv);
  int n1=100000,n2=100000;
  acl_thread_t t1,t2;
  cl_int status;

  // Optional argument to specify the problem size for each thread.
  // Sample usage: vector_operation_exe -n1=100000 -n2=100000
  if(options.has("n1")) {
    n1 = options.get<unsigned>("n1");
  }
  if(options.has("n2")) {
    n2 = options.get<unsigned>("n2");
  }

  // Initialize OpenCL.
  if(!init_opencl()) {
    return -1;
  }

  // First problem (thread1)
  Problem p1(n1,"vector_add");
  p1.init();
  // Second problem (thread2)
  // Can also be a different class
  Problem p2(n2,"vector_mult");
  p2.init();

  status = acl_thread_create(&t1, NULL, &start_helper, &p1);
  if (status != 0){
    printf("Can't create thread: %s\n", strerror(status));
  }
  else {
    printf("Thread1 created successfully\n");
  }

  status = acl_thread_create(&t2, NULL, &start_helper, &p2);
  if (status != 0){
    printf("Can't create thread: %s\n", strerror(status));
  }
  else {
    printf("Thread2 created successfully\n");
  }

  // Waiting for the threads to end...
  acl_thread_join(&t1);
  printf("Thread1 done\n");
  acl_thread_join(&t2);
  printf("Thread2 done\n");
  printf("Both threads completed.\n");

  p1.verify();
  p2.verify();
  printf("\nVerification: %s\n", (p1.pass && p2.pass) ? "PASS" : "FAIL");
  cleanup();
  return 0;
}

/////// HELPER FUNCTIONS ///////

// The helper function for starting the threads.
void* start_helper(void * arg) {
  Problem *p=(Problem*) arg;
  printf("Instantiating a new problem with args: N=%d kernel_name=%s \n",p->N,p->kernel_name);
  p->start();
  return NULL;
}

// Randomly generate a floating-point number between -10 and 10.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

#ifdef _WIN32   //Windows
int acl_thread_create (acl_thread_t *newthread, void *attr, void *(*start_routine) (void *), void *arg) {
  *newthread=CreateThread (NULL,0, (LPTHREAD_START_ROUTINE)start_routine,arg,DETACHED_PROCESS,NULL);
  attr=0;
  if (*newthread!=0) {
    return 0;
  } else {
    return GetLastError();
  }
}
int acl_thread_join(acl_thread_t *threadp){
  return WaitForMultipleObjects(1,threadp,TRUE,INFINITE);
}
#else         //Linux
// Thread type used for OS threads.
int acl_thread_create (acl_thread_t *newthread, void *attr, void *(*start_routine) (void *), void *arg) {
  return pthread_create(newthread,(const pthread_attr_t*) attr, start_routine,arg);
}
int acl_thread_join(acl_thread_t *threadp) {
  return pthread_join(*threadp,0);
}
#endif


// Initializes the Common OpenCL objects.
bool init_opencl() {
  cl_int status;

  printf("Initializing OpenCL\n");

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL device.
  device.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));
  printf("Platform: %s\n", getPlatformName(platform).c_str());
  printf("Using %d device(s)\n", num_devices);
  for(unsigned i = 0; i < num_devices; ++i) {
    printf("  %s\n", getDeviceName(device[i]).c_str());
  }

  // Create the context.
  context = clCreateContext(NULL, num_devices, device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the program for all device. Use the first device as the
  // representative device (assuming all device are of the same type).
  std::string binary_file = getBoardBinaryFile("vector_op", device[0]);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), device, num_devices);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  return true;
}

void cleanup() {
  if(program) {
    clReleaseProgram(program);
  }
  if(context) {
    clReleaseContext(context);
  }
}

/////// PROBLEM CLASS MEMBER FUNCTION IMPLEMENTATIONS ////////

void Problem::start()
{
  run();
  cleanup();
}

// Initialize the kernel, queue and data for the problem. Requires num_devices to be known.
void Problem::init() {
  cl_int status;

  if(num_devices == 0) {
    checkError(-1, "No devices");
  }

  // Create per-device objects.
  queue.reset(num_devices);
  kernel.reset(num_devices);
  n_per_device.reset(num_devices);
  input_a_buf.reset(num_devices);
  input_b_buf.reset(num_devices);
  output_buf.reset(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {
    // Command queue.
    queue[i] = clCreateCommandQueue(context, device[i], CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Kernel.
    kernel[i] = clCreateKernel(program, kernel_name, &status);
    checkError(status, "Failed to create kernel");

    // Determine the number of elements processed by this device.
    n_per_device[i] = N / num_devices; // number of elements handled by this device

    // Spread out the remainder of the elements over the first
    // N % num_devices.
    if(i < (N % num_devices)) {
      n_per_device[i]++;
    }

    // Input buffers.
    input_a_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
        n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input A");

    input_b_buf[i] = clCreateBuffer(context, CL_MEM_READ_ONLY,
        n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for input B");

    // Output buffer.
    output_buf[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        n_per_device[i] * sizeof(float), NULL, &status);
    checkError(status, "Failed to create buffer for output");
  }

  //initializing the data and calculating the reference output for comparison.
  input_a.reset(num_devices);
  input_b.reset(num_devices);
  output.reset(num_devices);
  ref_output.reset(num_devices);

  // Generate input vectors A and B and the reference output consisting
  // of a total of N elements.
  // We create separate arrays for each device so that each device has an
  // aligned buffer.
  for(unsigned i = 0; i < num_devices; ++i) {
    input_a[i].reset(n_per_device[i]);
    input_b[i].reset(n_per_device[i]);
    output[i].reset(n_per_device[i]);
    ref_output[i].reset(n_per_device[i]);

    for(unsigned j = 0; j < n_per_device[i]; ++j) {
      input_a[i][j] = rand_float();
      input_b[i][j] = rand_float();
      // Calculate the refrence output based on the current kernel.
      if(!strcmp(kernel_name,"vector_add")) {
        ref_output[i][j] = input_a[i][j] + input_b[i][j];
      }
      else if(!strcmp(kernel_name,"vector_mult")) {
        ref_output[i][j] = input_a[i][j] * input_b[i][j];
      }
    }
  }
}

void Problem::run() {
  cl_int status;

  const double start_time = getCurrentTimestamp();

  // Launch the problem for each device.
  scoped_array<cl_event> kernel_event(num_devices);
  scoped_array<cl_event> finish_event(num_devices);

  for(unsigned i = 0; i < num_devices; ++i) {
    // Transfer inputs to each device. Each of the host buffers supplied to
    // clEnqueueWriteBuffer here is already aligned to ensure that DMA is used
    // for the host-to-device transfer.
    cl_event write_event[2];
    status = clEnqueueWriteBuffer(queue[i], input_a_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(float), input_a[i], 0, NULL, &write_event[0]);
    checkError(status, "Failed to transfer input A");

    status = clEnqueueWriteBuffer(queue[i], input_b_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(float), input_b[i], 0, NULL, &write_event[1]);
    checkError(status, "Failed to transfer input B");

    // Set kernel arguments.
    unsigned argi = 0;

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_a_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &input_b_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    status = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &output_buf[i]);
    checkError(status, "Failed to set argument %d", argi - 1);

    // Enqueue kernel.
    // Use a global work size corresponding to the number of elements to add
    // for this device.
    //
    // We don't specify a local work size and let the runtime choose
    // (it'll choose to use one work-group with the same size as the global
    // work-size).
    //
    // Events are used to ensure that the kernel is not launched until
    // the writes to the input buffers have completed.
    const size_t global_work_size = n_per_device[i];
    printf("Launching for device %d (%d elements)\n", i, global_work_size);

    status = clEnqueueNDRangeKernel(queue[i], kernel[i], 1, NULL,
        &global_work_size, NULL, 2, write_event, &kernel_event[i]);
    checkError(status, "Failed to launch kernel");

    // Read the result. This the final operation.
    status = clEnqueueReadBuffer(queue[i], output_buf[i], CL_FALSE,
        0, n_per_device[i] * sizeof(float), output[i], 1, &kernel_event[i], &finish_event[i]);
    checkError(status, "Failed to read buffer");

    // Release local events.
    clReleaseEvent(write_event[0]);
    clReleaseEvent(write_event[1]);
  }

  // Wait for all devices to finish.
  clWaitForEvents(num_devices, finish_event);

  const double end_time = getCurrentTimestamp();

  // Wall-clock time taken.
  printf("\nTime: %0.3f ms\n", (end_time - start_time) * 1e3);

  // Get kernel times using the OpenCL event profiling API.
  for(unsigned i = 0; i < num_devices; ++i) {
    cl_ulong time_ns = getStartEndTime(kernel_event[i]);
    printf("Kernel time (device %d): %0.3f ms\n", i, double(time_ns) * 1e-6);
  }

  // Release all events.
  for(unsigned i = 0; i < num_devices; ++i) {
    clReleaseEvent(kernel_event[i]);
    clReleaseEvent(finish_event[i]);
  }
}

void Problem::verify() {
  // Verify results.
  pass = true;
  for(unsigned i = 0; i < num_devices && pass; ++i) {
    for(unsigned j = 0; j < n_per_device[i] && pass; ++j) {
      if(fabsf(output[i][j] - ref_output[i][j]) > 1.0e-5f) {
        printf("Failed verification @ device %d, index %d\nOutput: %f\nReference: %f\n",
            i, j, output[i][j], ref_output[i][j]);
        pass = false;
      }
    }
  }

}

// Free the resources allocated during initialization
void Problem::cleanup() {
  for(unsigned i = 0; i < num_devices; ++i) {
    if(kernel && kernel[i]) {
      clReleaseKernel(kernel[i]);
    }
    if(queue && queue[i]) {
      clReleaseCommandQueue(queue[i]);
    }
    if(input_a_buf && input_a_buf[i]) {
      clReleaseMemObject(input_a_buf[i]);
    }
    if(input_b_buf && input_b_buf[i]) {
      clReleaseMemObject(input_b_buf[i]);
    }
    if(output_buf && output_buf[i]) {
      clReleaseMemObject(output_buf[i]);
    }
  }
}

