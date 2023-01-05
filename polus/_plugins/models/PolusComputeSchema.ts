/* tslint:disable */
/* eslint-disable */
/**
/* This file was automatically generated from pydantic models by running pydantic2ts.
/* Do not modify it by hand - just update the pydantic models and then re-run the script
*/

export type GpuVendor = "none" | "amd" | "tpu" | "nvidia";
export type PluginInputType = "path" | "string" | "number" | "array" | "boolean";
export type PluginOutputType = "path";
export type PluginUIType =
  | "checkbox"
  | "color"
  | "date"
  | "email"
  | "number"
  | "password"
  | "radio"
  | "range"
  | "text"
  | "time";

export interface ConditionEntry {
  expression: string;
}
export interface Model {}
export interface PluginHardwareRequirements {
  coresMax?: string | number;
  coresMin?: string | number;
  cpuAVX?: boolean;
  cpuAVX2?: boolean;
  cpuMin?: string;
  gpu?: GpuVendor;
  gpuCount?: number;
  gpuDriverVersion?: string;
  gpuType?: string;
  outDirMax?: string | number;
  outDirMin?: string | number;
  ramMax?: string | number;
  ramMin?: string | number;
  tmpDirMax?: string | number;
  tmpDirMin?: string | number;
}
export interface PluginInput {
  format?: string;
  label?: string;
  name: string;
  required: boolean;
  type: PluginInputType;
  default?: string | number | boolean;
}
export interface PluginOutput {
  format?: string;
  label?: string;
  name: string;
  type: PluginOutputType;
}
export interface PluginSchema {
  author?: string;
  baseCommand?: string[];
  citation?: string;
  containerId: string;
  customInputs?: boolean;
  description: string;
  inputs: PluginInput[];
  institution?: string;
  name: string;
  outputs: PluginOutput[];
  pluginHardwareRequirements: PluginHardwareRequirements;
  repository?: string;
  title: string;
  ui: (PluginUIInput | PluginUIOutput)[];
  version: string;
  website?: string;
}
export interface PluginUIInput {
  bind?: string;
  condition?: Validator[] | string;
  default?: string | number | boolean;
  description?: string;
  fieldset?: string[];
  hidden?: boolean;
  key: string;
  title: string;
  type: PluginUIType;
}
export interface Validator {
  then?: ThenEntry[];
  validator?: ConditionEntry[];
}
export interface ThenEntry {
  action: string;
  input: string;
  value: string;
}
export interface PluginUIOutput {
  description: string;
  format?: string;
  name: string;
  type: PluginUIType;
  website?: string;
}
