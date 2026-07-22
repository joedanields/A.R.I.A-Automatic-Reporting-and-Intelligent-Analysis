/**
 * A.R.I.A. Shared Types
 *
 * Common types used across frontend components.
 */

/** Character offset span into the transcript */
export interface SourceSpan {
  start_char: number;
  end_char: number;
}

/** Provenance value for clinical data */
export type ProvenanceValue = "heard" | "retrieved" | "inferred";

/** Provenance tag attached to entities, codes, and SOAP sections */
export interface ProvenanceTag {
  field: string;
  value: string;
  provenance: ProvenanceValue;
  source_span?: SourceSpan;
}

/** ICD code with provenance */
export interface IcdCode {
  code: string;
  description: string;
  confidence?: string;
  provenance?: ProvenanceValue;
  source_span?: SourceSpan;
}

/** SOAP section with provenance */
export interface SoapSection {
  title: string;
  text: string;
  codes?: IcdCode[];
  provenance?: ProvenanceValue;
  source_span?: SourceSpan;
}

/** Validation result from F3 validator */
export interface ValidationResult {
  grounded: boolean;
  flags: string[];
  ungrounded_claims: string[];
}
