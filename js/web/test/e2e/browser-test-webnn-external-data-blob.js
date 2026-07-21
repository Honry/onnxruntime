// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// WebNN is gated behind a Chrome feature flag and is not available on every runner. Probe for it
// and skip (report as pending) instead of failing when it is unavailable.
async function isWebNNAvailable() {
  if (typeof navigator === 'undefined' || !navigator.ml || typeof navigator.ml.createContext !== 'function') {
    return false;
  }
  try {
    const context = await navigator.ml.createContext();
    return !!context;
  } catch (e) {
    return false;
  }
}

function assertResult(fetches) {
  const Y = fetches.Y;
  assert(Y instanceof ort.Tensor);
  assert(Y.dims.length === 2 && Y.dims[0] === 2 && Y.dims[1] === 3);
  assert(Y.data[0] === 1);
  assert(Y.data[1] === 1);
  assert(Y.data[2] === 0);
  assert(Y.data[3] === 0);
  assert(Y.data[4] === 0);
  assert(Y.data[5] === 0);
}

it('Browser E2E testing - WebNN backend with external data as Blob', async function () {
  if (!(await isWebNNAvailable())) {
    this.skip();
    return;
  }

  // Supplying external data as a Blob. In a JSPI build the WebNN EP reads each initializer's byte
  // range from the Blob on demand during session initialization instead of materializing the whole
  // file in memory at once; in the Asyncify build the Blob is fully materialized first (fallback).
  // Either way the result must be correct.
  const blob = await (await fetch('./model_with_orig_ext_data.bin')).blob();
  const session = await ort.InferenceSession.create('./model_with_orig_ext_data.onnx', {
    executionProviders: ['webnn'],
    externalData: [{ data: blob, path: 'model_with_orig_ext_data.bin' }],
  });

  const fetches = await session.run({ X: new ort.Tensor('float32', [1, 1], [1, 2]) });
  assertResult(fetches);
});

it('Browser E2E testing - WebNN backend with external data as Uint8Array', async function () {
  if (!(await isWebNNAvailable())) {
    this.skip();
    return;
  }

  // Supplying external data as a whole-file Uint8Array (the pre-existing path) must keep working.
  const data = new Uint8Array(await (await fetch('./model_with_orig_ext_data.bin')).arrayBuffer());
  const session = await ort.InferenceSession.create('./model_with_orig_ext_data.onnx', {
    executionProviders: ['webnn'],
    externalData: [{ data, path: 'model_with_orig_ext_data.bin' }],
  });

  const fetches = await session.run({ X: new ort.Tensor('float32', [1, 1], [1, 2]) });
  assertResult(fetches);
});
