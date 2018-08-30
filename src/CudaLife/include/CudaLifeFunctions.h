#pragma once

#include <cstddef>

namespace mf {

	typedef unsigned char uint8_t;
	typedef unsigned short ushort;
	typedef unsigned int uint;


	extern "C" bool runSimpleLifeKernel(uint8_t*& d_lifeData, uint8_t*& d_lifeDataBuffer, size_t worldWidth,
		size_t worldHeight, size_t iterationsCount, ushort threadsCount);


	extern "C" void runPrecompute6x3EvaluationTableKernel(uint8_t* d_lookupTable);

	extern "C" void runBitLifeEncodeKernel(const uint8_t* d_lifeData, uint worldWidth, uint worldHeight,
		uint8_t* d_encodedLife);

	extern "C" void runBitLifeDecodeKernel(const uint8_t* d_encodedLife, uint worldWidth, uint worldHeight,
		uint8_t* d_lifeData);

	extern "C" bool runBitLifeKernel(uint8_t*& d_lifeData, uint8_t*& d_lifeDataBuffer, const uint8_t* d_lookupTable,
		size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount, uint bytesPerThread,
		bool useBigChunks);


	extern "C" void runDisplayLifeKernel(const uint8_t* d_lifeData, size_t worldWidth, size_t worldHeight,
		uchar4 *destination, int destWidth, int detHeight, int displacementX, int displacementY, int zoom,
		bool simulateColors, bool cyclic, bool bitLife);

}
