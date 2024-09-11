// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

export type ConstantId = number;

/**
 * Manages ConstantId to MLConstant mapping.
 */
export interface ConstantManager {
  /**
   * Reserve a new ConstantId.
   */
  reserveConstantId(constantId: ConstantId): void;
  /**
   * Release a ConstantId.
   */
  releaseConstantId(constantId: ConstantId): void;
  /**
   * Upload data to a MLConstant.
   */
  upload(constantId: ConstantId, data: Uint8Array): void;
  /**
   * Download data associated with the provided ConstantId.
   */
  download(constantId: ConstantId): ArrayBuffer | null | undefined;
}

class ConstantManagerImpl implements ConstantManager {
  private constantsById = new Map<ConstantId, Uint8Array | null>();

  public reserveConstantId(constantId: ConstantId): void {
    this.constantsById.set(constantId, null);
  }

  public releaseConstantId(constantId: ConstantId): void {
    this.constantsById.delete(constantId);
  }

  public upload(constantId: ConstantId, data: Uint8Array): void {
    if (!this.constantsById.has(constantId)) {
      throw new Error(`Failed to upload constant: the constant with id ${constantId} does not exist.`);
    } else if (this.constantsById.get(constantId) !== null) {
      throw new Error(`Failed to upload constant: the constant with id ${constantId} has already been uploaded.`);
    } else {
      this.constantsById.set(constantId, new Uint8Array(data));
    }
    console.log('constant-manager.ts: uploaded ', constantId);
  }

  public download(constantId: ConstantId): ArrayBuffer | null | undefined {
    if (!this.constantsById.has(constantId)) {
      throw new Error(`Failed to download constant: the constant with id ${constantId} does not exist.`);
    } else if (this.constantsById.get(constantId) === null) {
      throw new Error(`Failed to download constant: the constant with id ${constantId} has no data.`);
    } else {
      const data = this.constantsById.get(constantId);
      return data?.buffer;
    }
  }
}

export const createConstantManager = (...args: ConstructorParameters<typeof ConstantManagerImpl>): ConstantManager =>
  new ConstantManagerImpl(...args);
