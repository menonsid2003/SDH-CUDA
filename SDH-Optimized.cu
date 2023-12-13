/* ==================================================================
        CPU implementation: Yicheng Tu (ytu@cse.usf.edu)
        GPU implementation + comparison: Sidharth Menon
        Optimized SDH algorithm implementation for 3D data
        To compile:  /apps/GPU_course/runScript.sh proj2-menons.cu [param 1 eg:
        10000] [param 2 eg:500] [param 3 eg:32] in the GAIVI machines
   ==================================================================
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


#define BOX_SIZE 23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
  double x_pos;
  double y_pos;
  double z_pos;
} atom;

typedef struct hist_entry {
  unsigned long long int
      d_cnt; /* need a long long type as the count might be huge */
} bucket;

cudaEvent_t TIMING_START_EVENT, TIMING_STOP_EVENT;
float TIMING_ELAPSED_TIME;

void TIMING_START(){
    cudaEventCreate (&TIMING_START_EVENT);
    cudaEventCreate (&TIMING_STOP_EVENT);
    cudaEventRecord (TIMING_START_EVENT, 0);
}


void TIMING_STOP(){
    cudaEventRecord (TIMING_STOP_EVENT, 0);
    cudaEventSynchronize (TIMING_STOP_EVENT);
    cudaEventElapsedTime (&TIMING_ELAPSED_TIME, TIMING_START_EVENT, TIMING_STOP_EVENT);
    cudaEventDestroy (TIMING_START_EVENT);
    cudaEventDestroy (TIMING_STOP_EVENT);
}

void TIMING_PRINT(){
    printf("Timing is: %f ms\n", TIMING_ELAPSED_TIME);
}

bucket* histogramCPU; /* list of all buckets in the histogram   */
bucket* histogramGPU;
bucket* histogramCopyHelp;
long long PDH_acnt; /* total number of data points            */
int num_buckets;    /* total number of buckets in the histogram */
double PDH_res;     /* value of w                             */
atom* atomListCPU;  /* list of all data points                */
atom* atomListGPU;

int blockSize;

/* These are for an old way of tracking time */
struct timezone Idunno;
struct timeval endTime;

// find distance of two points in the atom_list, only called for CPU calculation
__host__ double p2p_distance(atom* al, int ind1, int ind2) {
  double x1 = al[ind1].x_pos;
  double x2 = al[ind2].x_pos;
  double y1 = al[ind1].y_pos;
  double y2 = al[ind2].y_pos;
  double z1 = al[ind1].z_pos;
  double z2 = al[ind2].z_pos;

  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
              (z1 - z2) * (z1 - z2));
}

// slightly different implementation of p2p_distance that takes two atoms
// as input instead of the list and indices to utilize shared memory and data from two different atomLists easier
__device__ double p2p_distanceGPU(atom& a, atom& b) {
  double x1 = a.x_pos;
  double x2 = b.x_pos;
  double y1 = a.y_pos;
  double y2 = b.y_pos;
  double z1 = a.z_pos;
  double z2 = b.z_pos;

  return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) +
              (z1 - z2) * (z1 - z2));
}

// brute-force SDH solution in a single CPU thread
int PDH_baseline() {
  int i, j, h_pos;
  double dist;

  for (i = 0; i < PDH_acnt; i++) {
    for (j = i + 1; j < PDH_acnt; j++) {
      dist = p2p_distance(atomListCPU, i, j);
      h_pos = (int)(dist / PDH_res);
      histogramCPU[h_pos].d_cnt++;
    }
  }
  return 0;
}

// GPU implementation
__global__ void PDH_baseline_gpu(bucket* histogramGPU, atom* atomListGPU, double PDH_res, long long PDH_acnt, int num_buckets) {
  int i, j, pos;
  double dist;

  int bdim = blockDim.x;    // Block dimension
  int tid = threadIdx.x;    // Thread index within the block
  int bid = blockIdx.x;     // Block index within the grid

  // Shared memory to hold input data
  extern __shared__ atom sharedAtomList[];
  // Current atom for this particular thread
  atom curr;

  // Shared Histogram GPU to count within the thread, also implicitly declared as shared due to using sharedAtomList
  bucket* localHistogram = (bucket*)&sharedAtomList[bdim];

  // Calculate the global thread ID
  long long id = blockDim.x * blockIdx.x + threadIdx.x;

  // Initialize localHistogram in shared memory for each thread
  for (i = tid; i < num_buckets; i = i + bdim) {
    localHistogram[i].d_cnt = 0;
  }
  //__syncthreads();

  // Current atom of this particular thread
  curr = atomListGPU[id];

  // Loop through different blocks of atoms
  for (i = bid + 1; i < gridDim.x; i++) {
    // Copies an atom to shared memory for the current thread 
    sharedAtomList[tid] = atomListGPU[tid + (i * bdim)];
    __syncthreads();
    // A pre-loop check to see if the starting index is within the valid range (PDH_acnt)
    if ((i * bdim) < PDH_acnt) {
      for (j = 0; j < bdim; j++) {
        if (j + (i * bdim) < PDH_acnt) {  // One more safety check to ensure atom with offset index is within the valid range
          // Calculate the pairwise distance between the current atom and an atom from shared memory
          dist = p2p_distanceGPU(curr, sharedAtomList[j]);
          // Calculate the position (or bin) in the histogram where the count should be incremented based on the computed distance
          pos = (int)(dist / PDH_res);
          // Atomic operation to increment the count in the shared histogram at the specified position
          atomicAdd(&(localHistogram[pos].d_cnt), 1);
        }
      }
    }
    __syncthreads();
  }

  // Restore the current atom in sharedAtomList
  sharedAtomList[tid] = curr;
  //__syncthreads();

  int bdimHalf;
  // Determine half of the block dimension (for use in a loop)
  if (bdim % 2 == 0 && tid >= bdim / 2) {
    bdimHalf = bdim / 2 - 1;
  } else {
    bdimHalf = bdim / 2;
  }

  // Continue processing for the second half of the block
  if (id < PDH_acnt) {
    for (j = 0; j < bdimHalf; j++) {
      int index = (tid + j + 1) % bdim;
      if (index + (bdim * bid) < PDH_acnt) {
        // Calculate the pairwise distance between the current atom and another atom from shared memory
        dist = p2p_distanceGPU(curr, sharedAtomList[index]);
        // Calculate the position (or bin) in the histogram for the count increment
        pos = (int)(dist / PDH_res);
        // Atomic operation to increment the count in the shared histogram at the specified position
        atomicAdd(&(localHistogram[pos].d_cnt), 1);
      }
    }
  }
  __syncthreads();
  
  // Reduce Shared Histogram in Device back into parameter Histogram in Host
  for (i = tid; i < num_buckets; i = i + bdim) {
    atomicAdd(&(histogramGPU[i].d_cnt), (localHistogram[i].d_cnt));
  }
}


// set a checkpoint and show the (natural) running time in seconds
void report_running_time(struct timeval startTime) {
  struct timeval endTime;
  long sec_diff, usec_diff;
  gettimeofday(&endTime, &Idunno);
  sec_diff = endTime.tv_sec - startTime.tv_sec;
  usec_diff = endTime.tv_usec - startTime.tv_usec;
  if (usec_diff < 0) {
    sec_diff--;
    usec_diff += 1000000;
  }
  float elaps = (sec_diff * 1000) + (usec_diff / 1000.0); // Calculate time in milliseconds
  printf("Running time for CPU version: %.2f ms\n", elaps); // Print time in milliseconds
}


// print the counts in all buckets of the histogram
void output_histogram(bucket* histo) {
  int i;
  long long total_cnt = 0;
  for (i = 0; i < num_buckets; i++) {
    if (i % 5 == 0) /* we print 5 buckets in a row */
      printf("\n%02d: ", i);
    printf("%15lld ", histo[i].d_cnt);
    total_cnt += histo[i].d_cnt;
    /* we also want to make sure the total distance count is correct */
    if (i == num_buckets - 1)
      printf("\nT:%lld \n", total_cnt);
    else
      printf("| ");
  }
}

// similar to the output_histogram but outputs the difference between two
// histograms
void histogramComp(bucket* histo1, bucket* histo2) {
  int i;
  long long total_cnt = 0;
  printf("CPU histogram - GPU histogram:\n");
  for (i = 0; i < num_buckets; i++) {
    if (i % 5 == 0) /* we print 5 buckets in a row */
      printf("\n%02d: ", i);
    printf("%15lld ", histo1[i].d_cnt - histo2[i].d_cnt);
    total_cnt += histo1[i].d_cnt;
    /* we also want to make sure the total distance count is correct */
    if (i == num_buckets - 1)
      printf("\nT:%lld \n", total_cnt);
    else
      printf("| ");
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    // Check if the number of arguments is correct
    printf("\n-----------------------------------------------------------------------------------------------------\n");
    printf("Usage: %s <PDH_acnt> <PDH_res> <blockSize>\n", argv[0]);
    printf("--------------------------------------------EXITING--------------------------------------------------\n\n");
    return 1; // Return with an error code
  }

  int i;

  PDH_acnt = atoi(argv[1]);
  PDH_res = atof(argv[2]);
  blockSize = atoi(argv[3]);

  if (PDH_acnt <= 0 || PDH_res <= 0 || blockSize <= 0) {
    // Check if the provided arguments are valid
    printf("\n-----------------------------------------------------------------------------------------------------\n");
    printf("Error: All arguments (PDH_acnt, PDH_res, blockSize) must be positive.\n");
    printf("--------------------------------------------EXITING--------------------------------------------------\n\n");
    return 1; // Return with an error code
  }

  num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
  histogramCPU = (bucket*)malloc(sizeof(bucket) * num_buckets);
  atomListCPU = (atom*)malloc(sizeof(atom) * PDH_acnt);

  srand(1);
  /* generate data following a uniform distribution */
  for (i = 0; i < PDH_acnt; i++) {
    atomListCPU[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    atomListCPU[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    atomListCPU[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
  }

  // CPU Start
  /* start counting time */
  struct timeval startTime;
  gettimeofday(&startTime, &Idunno);

  /* call CPU single thread version to compute the histogram */
  PDH_baseline();
  printf("\n-----------------------------------------------------------------------------------------------------\n");
  printf("CPU implementation:\n");
  output_histogram(histogramCPU);

  report_running_time(startTime);
  printf("-----------------------------------------------------------------------------------------------------\n");
  // CPU Done

  // GPU Start

  // If these are initialized as NULL instead like in project 1, the
  // calculations were massively incorrect for some reason. Likely due to some
  // size mismatching in dynamic shared memory.
  histogramGPU = (bucket*)malloc(sizeof(bucket) * num_buckets);
  histogramCopyHelp = (bucket*)malloc(sizeof(bucket) * num_buckets);

  atomListGPU = (atom*)malloc(sizeof(atom) * PDH_acnt);
  cudaMalloc((void**)&atomListGPU, sizeof(atom) * PDH_acnt);

  cudaMemcpy(atomListGPU, atomListCPU, sizeof(atom) * PDH_acnt,
             cudaMemcpyHostToDevice);

  // The atomListGPU CUDA initialization needs to be before the histogram
  // initialization. Otherwise, again the SDH is completely broken. Not sure why,
  // but this order of commands fixes the error.
  cudaMalloc((void**)&histogramGPU, sizeof(bucket) * num_buckets);
  cudaMemcpy(histogramGPU, histogramCopyHelp, sizeof(bucket) * num_buckets, cudaMemcpyHostToDevice);

  TIMING_START();

  int blockCount = ceil(PDH_acnt / (float)blockSize);
  PDH_baseline_gpu<<<blockCount, blockSize, sizeof(bucket) * num_buckets + sizeof(atom) * blockSize>>>(
      histogramGPU, atomListGPU, PDH_res, PDH_acnt, num_buckets);

  cudaMemcpy(histogramCopyHelp, histogramGPU, sizeof(bucket) * num_buckets, cudaMemcpyDeviceToHost);

  /* print out the histogram */
  printf("Project 2 GPU implementation:\n");
  output_histogram(histogramCopyHelp);

  TIMING_STOP();
  TIMING_PRINT();

  printf("-----------------------------------------------------------------------------------------------------\n");
  // CPU - GPU time comparison print.
  histogramComp(histogramCPU, histogramCopyHelp);

  printf("-----------------------------------------------------------------------------------------------------\n");
  // freeing all allocated memory in CPU and GPU and resetting the cuda device
  free(histogramCPU);
  cudaFree(histogramGPU);

  free(histogramCopyHelp);
  cudaFree(histogramCopyHelp);

  free(atomListCPU);
  cudaFree(atomListGPU);

  cudaDeviceReset();
  return 0;
}
