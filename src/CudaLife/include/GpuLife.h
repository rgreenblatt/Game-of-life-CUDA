#pragma once
#include "OpenGlCudaHelper.h"
#include "CudaLifeFunctions.h"
#include <cstddef>
#include <stdint.h>
#include <random>
#include <array>
#include <algorithm>
#include <functional>
#include <cassert>
#include <cstring>
//#include "CpuLife.h"

namespace mf {

	template<typename NoCppFileNeeded = int>
	class TGpuLife {

	private:
		uint8_t* d_lifeData;
		uint8_t* d_lifeDataBuffer;

		uint8_t* d_encLifeData;
		uint8_t* d_encLifeDataBuffer;

		uint8_t* d_lookupTable;

		/// If unsuccessful allocation occurs, size is saved and never tried to allocate again to avoid many
		/// unsuccessful allocations in the row.
		size_t m_unsuccessAllocSize;


		/// Current width of world.
		size_t m_worldWidth;
		/// Current height of world.
		size_t m_worldHeight;

		std::mt19937 m_randGen;
		
        void initRand(uint8_t* data, size_t length, uint mask, bool useBetterRandom) {
			if (useBetterRandom) {
				for (size_t i = 0; i < length; ++i) {
					data[i] = uint8_t(m_randGen() & mask);
				}
			}
			else {
				for (size_t i = 0; i < length; ++i) {
					data[i] = uint8_t(rand() & mask);
				}
			}
		}
	public:
		TGpuLife()
			: d_lifeData(nullptr)
			, d_lifeDataBuffer(nullptr)
			, d_encLifeData(nullptr)
			, d_encLifeDataBuffer(nullptr)
			, d_lookupTable(nullptr)
			, m_unsuccessAllocSize(std::numeric_limits<size_t>::max())
			, m_worldWidth(0)
			, m_worldHeight(0)
		{
		    std::array<int, std::mt19937::state_size> seed_data;
		    std::random_device r;
		    std::generate_n(seed_data.begin(), seed_data.size(), std::ref(r));
		    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));

            m_randGen = std::mt19937(seq);
        }

		~TGpuLife() {
			freeBuffers();

			checkCudaErrors(cudaFree(d_lookupTable));
			d_lookupTable = nullptr;
		}

		const uint8_t* getLifeData() const {
			return d_lifeData;
		}

		uint8_t* lifeData() {
			return d_lifeData;
		}

		const uint8_t* getBpcLifeData() const {
			return d_encLifeData;
		}

		uint8_t* bpcLifeData() {
			return d_encLifeData;
		}

		const uint8_t* getLookupTable() {
			if (d_lookupTable == nullptr) {
				computeLookupTable();
			}
			return d_lookupTable;
		}

		/// Returns true if buffers for given life algorithm type are allocated and ready for use.
		bool areBuffersAllocated(bool bitLife) const {
			if (bitLife) {
				return d_encLifeData != nullptr && d_encLifeDataBuffer != nullptr;
			}
			else {
				return d_lifeData != nullptr && d_lifeDataBuffer != nullptr;
			}
		}

		/// Frees all buffers and allocated buffers necessary for given algorithm type.
		bool allocBuffers(bool bitLife) {
			freeBuffers();

			if (bitLife) {
				size_t worldSize = (m_worldWidth / 8) * m_worldHeight;

				if (worldSize >= m_unsuccessAllocSize) {
					return false;
				}

				if (cudaMalloc(&d_encLifeData, worldSize) || cudaMalloc(&d_encLifeDataBuffer, worldSize)) {
					// Allocation failed.
					freeBuffers();
					m_unsuccessAllocSize = worldSize;
					return false;
				}
			}
			else {
				size_t worldSize = m_worldWidth * m_worldHeight;

				if (worldSize >= m_unsuccessAllocSize) {
					return false;
				}

				if (cudaMalloc(&d_lifeData, worldSize) || cudaMalloc(&d_lifeDataBuffer, worldSize)) {
					// Allocation failed.
					freeBuffers();
					m_unsuccessAllocSize = worldSize;
					return false;
				}
			}

			return true;
		}

		/// Frees all dynamically allocated buffers (expect lookup table).
		void freeBuffers() {
			checkCudaErrors(cudaFree(d_lifeData));
			d_lifeData = nullptr;

			checkCudaErrors(cudaFree(d_lifeDataBuffer));
			d_lifeDataBuffer = nullptr;

			checkCudaErrors(cudaFree(d_encLifeData));
			d_encLifeData = nullptr;

			checkCudaErrors(cudaFree(d_encLifeDataBuffer));
			d_encLifeDataBuffer = nullptr;

			// Do not free lookup table.
		}

		/// Resizes the world and frees old buffers.
		/// Do not allocates new buffers (lazy allocation, buffers are allocated when needed).
		void resize(size_t newWidth, size_t newHeight) {
			freeBuffers();

			m_worldWidth = newWidth;
			m_worldHeight = newHeight;
		}

		/// Initializes necessary arrays with values. Uses CPU life instance for random data generation.
		void initThis(bool bitLife, bool useBetterRandom) {
			std::vector<uint8_t> encData;

			size_t worldSize = (m_worldWidth / 8) * m_worldHeight;
			if (bitLife) {
				encData.resize(worldSize);  // Potential bad_alloc.
				initRand(&encData[0], worldSize, 0xFF, useBetterRandom);

				checkCudaErrors(cudaMemcpy(d_encLifeData, &encData[0], worldSize, cudaMemcpyHostToDevice));
			}
			else {
				std::vector<uint8_t> encData;
				encData.resize(worldSize);  // Potential bad_alloc.
				initRand(&encData[0], worldSize, 0x1, useBetterRandom);

				checkCudaErrors(cudaMemcpy(d_lifeData, &encData[0], worldSize, cudaMemcpyHostToDevice));
			}
		}

		bool iterate(size_t lifeIteratinos, bool useLookupTable, ushort threadsCount, bool bitLife,
			uint bitLifeBytesPerTrhead, bool useBigChunks) {

			if (bitLife) {
				return runBitLifeKernel(d_encLifeData, d_encLifeDataBuffer, useLookupTable ? d_lookupTable : nullptr,
					m_worldWidth, m_worldHeight, lifeIteratinos, threadsCount, bitLifeBytesPerTrhead, useBigChunks);
			}
			else {
				return runSimpleLifeKernel(d_lifeData, d_lifeDataBuffer, m_worldWidth, m_worldHeight,
					lifeIteratinos, threadsCount);
			}
		}


	private:
		void computeLookupTable() {
			checkCudaErrors(cudaMalloc((void**)&d_lookupTable, 1 << (6 * 3)));
			runPrecompute6x3EvaluationTableKernel(d_lookupTable);
		}

	};

	typedef TGpuLife<> GpuLife;

}
