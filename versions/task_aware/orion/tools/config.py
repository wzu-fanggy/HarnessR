"""Submission registry. Formal protocols further restrict this tool set."""
from tools.file_operate import write_file, read_file
from tools.todo_tools import read_todos, write_todos
from tools.rag.kg_rag_tools import (aggregate_evidence_by_entities, explore_entity_neighborhood,
 resolve_literature_papers, search_paper_chunks, search_kg_rag_hybrid, search_literature_multiview)
from tools.kg_query import kg_query
from tools.electrolyte_design.conductivity_tool import predict_formulation_conductivity
from tools.electrolyte_design.candidate_space import generate_candidate_space
from tools.electrolyte_design.candidate_batch_tools import evaluate_candidate_batch, inspect_candidate_artifact, compare_conductivity_predictions
from tools.electrolyte_design.temperature_fit_tool import fit_conductivity_temperature_law
from tools.electrolyte_design.robust_ranking_tool import rank_formulations_under_uncertainty
from tools.electrolyte_design.properties import extract_numeric_properties
from orion.core.tasks import TASK_TOOLS
main_tools=[write_file,read_file,read_todos,write_todos,kg_query,search_kg_rag_hybrid,
 resolve_literature_papers,search_paper_chunks,search_literature_multiview,
 explore_entity_neighborhood,aggregate_evidence_by_entities,predict_formulation_conductivity,
 generate_candidate_space,evaluate_candidate_batch,inspect_candidate_artifact,
 compare_conductivity_predictions,fit_conductivity_temperature_law,
 rank_formulations_under_uncertainty,extract_numeric_properties,*TASK_TOOLS]
sub_tools=[]

# Additional tools present in the formal screening/recommendation traces.
from tools.skill_tools import load_skill
from tools.electrolyte_design.screen import screen_formulations
from tools.electrolyte_design.synthesis import synthesize_design_report
from tools.electrolyte_design.enrichment import enrich_formulation_evidence
from tools.electrolyte_design.formulate import query_formulation_space
main_tools += [load_skill,screen_formulations,synthesize_design_report,enrich_formulation_evidence,query_formulation_space]
