// tile_descriptor.cuh
// Tile status and descriptor types for decoupled lookback algorithms

#pragma once

#include <cuda_runtime.h>

// ============================================================================
// TILE STATUS
// ============================================================================
// Three-state system for decoupled lookback:
// - INVALID:   Tile has not yet been processed
// - AGGREGATE: Tile has computed its local sum but not its prefix
// - PREFIX:    Tile has computed its complete prefix sum

enum class TileStatus : int {
    INVALID   = 0,
    AGGREGATE = 1,
    PREFIX    = 2
};

// ============================================================================
// TILE DESCRIPTOR
// ============================================================================
// Union allows atomic 64-bit operations on combined status + value.
// Both fields are updated atomically together, preventing torn reads.

union TileDescriptor {
    unsigned long long int raw;  // For atomic operations
    struct {
        int value;               // Aggregate or prefix sum
        TileStatus status;       // Current tile state
    };
};

static_assert(sizeof(TileDescriptor) == 8, "TileDescriptor must be 8 bytes for atomic ops");