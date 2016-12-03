#include "CudaLoader.h"
#include "CudaDef.h"

void CudaLoader::loadToDevice() {
	//copy image data to global memory
	CUDA_CHECK_ERROR(cudaMalloc((void**)&pd_image, sizeof(glm::vec3)*p_image->numPixels));
	CUDA_CHECK_ERROR(cudaMemcpy(&pd_image, p_image->data, 1, cudaMemcpyHostToDevice));

	CUD

}
