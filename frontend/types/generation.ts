/**
 * Generation types for Voice Generation API
 */

/**
 * Inference phase enum matching backend InferencePhase
 * Note: Backend currently returns lowercase values
 */
export enum InferencePhase {
  PENDING = 'pending',
  PREPROCESSING = 'preprocessing',
  INFERENCING = 'inferencing',
  SAVING_AUDIO = 'saving_audio',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

/**
 * Model dtype options
 */
export type ModelDtype = 'bf16' | 'float8_e4m3fn';

/**
 * Generation metadata from backend
 */
export interface Generation {
  request_id: string;
  session_id: string;
  status: InferencePhase;
  output_filename: string | null;
  percentage: number | null;
  model_dtype: ModelDtype;
  cfg_scale: number | null;
  attn_implementation: string | null;
  seeds: number;
  details: Record<string, any>;
  created_at: string;
  updated_at: string;
}

/**
 * Request body for creating generation
 */
export interface CreateGenerationRequest {
  dialog_session_id: string;
  seeds?: number;
  cfg_scale?: number;
  model_dtype?: ModelDtype;
  attn_implementation?: string;
}

/**
 * Response from POST /generations
 */
export interface CreateGenerationResponse {
  message: string;
  request_id: string;
  generation: Generation;
}

/**
 * Response from GET /generations/current
 * Note: generation is null when no active generation (200 response, not an error)
 */
export interface CurrentGenerationResponse {
  message: string;
  generation: Generation | null;
}

/**
 * Response from GET /generations
 */
export interface ListGenerationsResponse {
  generations: Generation[];
  count: number;
}

/**
 * Response from GET /generations/:request_id
 */
export interface GetGenerationResponse {
  generation: Generation;
}
