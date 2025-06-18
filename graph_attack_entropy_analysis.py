import json
import numpy as np
import time
import torch
import torch.nn.functional as F
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
import logging
from tqdm import tqdm
from collections import defaultdict
import traceback
# Import from your existing code
from connectivity_disruption_proposed_attack_model import (
    start_attack_proposed_model, 
    get_dataset_from_deeprobust,
    ProposedAttack,
    get_device
)

from utils import get_target_node_list_v2


# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_to_json_serializable(obj):
    """
    Convert numpy/torch objects to JSON serializable format
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist() if obj.numel() > 1 else obj.cpu().detach().numpy().item()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, bool):
        return bool(obj)
    elif hasattr(obj, 'item'):  # For single-element tensors
        return obj.item()
    else:
        return obj

class ConnectivityDisruptionExperiment:
    def __init__(self, surrogate_model='gcn', dataset='cora', defense_model='gcn'):
        self.surrogate_model = surrogate_model
        self.dataset = dataset
        self.defense_model = defense_model
        self.device = get_device()
        
        # Load dataset and get target nodes
        self.data = get_dataset_from_deeprobust(dataset=dataset)
        self.target_nodes = get_target_node_list_v2(self.data)
        
        # Load important edge list
        with open('./important_edge_list.json', 'r') as json_file:
            important_edge_list_dict = json.load(json_file)
        self.important_edge_list = [tuple(item) for item in important_edge_list_dict[dataset]]
        
        # Initialize attack model
        self.attack_model = ProposedAttack(
            surrogate_model, dataset, defense_model, 
            self.important_edge_list
        )
        
        # Results storage
        self.experiment_results = []
        self.misclassified_results = []  # Only properly misclassified nodes
        self.right_classified_results = []  # Only properly right classified nodes
        
    def compute_entropy(self, probabilities):
        """
        Compute classification entropy H(w) = -sum(p_i * log(p_i))
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1.0)
        return -np.sum(probabilities * np.log(probabilities))
    
    def compute_kl_divergence(self, p_original, p_perturbed):
        """
        Compute KL divergence KL(p||q) = sum(p_i * log(p_i/q_i))
        """
        epsilon = 1e-10
        p_original = np.clip(p_original, epsilon, 1.0)
        p_perturbed = np.clip(p_perturbed, epsilon, 1.0)
        
        return np.sum(p_original * np.log(p_original / p_perturbed))
    
    def get_node_probabilities_before_attack(self, target_node):
        """
        Get classification probabilities for target node before attack
        """
        # Get original model output
        surrogate_output = self.attack_model.surrogate_model_output
        probs_before = F.softmax(surrogate_output[target_node], dim=0)
        return probs_before.cpu().detach().numpy()
    
    def get_node_probabilities_after_attack(self, target_node, modified_adj):
        """
        Get classification probabilities for target node after attack
        """
        accuracy, pred_class, actual_class, probs_after = self.attack_model.predict(
            modified_adj, target_node
        )
        return probs_after
    
    def is_properly_misclassified(self, actual_class, predicted_before, predicted_after):
        """
        Check if node is properly misclassified:
        1. Originally correctly classified (actual == predicted_before)
        2. After attack wrongly classified (actual != predicted_after)
        """
        originally_correct = (actual_class == predicted_before)
        after_attack_wrong = (actual_class != predicted_after)
        
        return originally_correct and after_attack_wrong
    
    def is_properly_right_classified(self, actual_class, predicted_before, predicted_after):
        """
        Check if node remains properly right classified:
        1. Originally correctly classified (actual == predicted_before)
        2. After attack still correctly classified (actual == predicted_after)
        """
        originally_correct = (actual_class == predicted_before)
        after_attack_correct = (actual_class == predicted_after)
        
        return originally_correct and after_attack_correct
    
    def run_single_node_experiment(self, target_node, budget=5):
        """
        Run experiment on a single node
        """
        try:
            logger.info(f"Running experiment on node {target_node}")
            
            # Get probabilities before attack
            probs_before = self.get_node_probabilities_before_attack(target_node)
            entropy_before = self.compute_entropy(probs_before)
            
            # Get actual and predicted class before attack
            actual_class_before = int(self.data.labels[target_node])
            predicted_class_before = int(np.argmax(probs_before))
            
            # Perform attack
            modified_adj, attack_history = self.attack_model.attack(
                target_node=target_node, 
                n_perturbations=budget
            )
            
            # Get probabilities after attack
            accuracy_after, predicted_class_after, actual_class_after, probs_after = self.attack_model.predict(
                modified_adj, target_node
            )
            
            # Convert to proper types
            predicted_class_after = int(predicted_class_after)
            entropy_after = self.compute_entropy(probs_after)
            
            # Compute changes
            delta_entropy = entropy_after - entropy_before
            kl_divergence = self.compute_kl_divergence(probs_before, probs_after)
            
            # Check classification status
            properly_misclassified = self.is_properly_misclassified(
                actual_class_before, predicted_class_before, predicted_class_after
            )
            
            properly_right_classified = self.is_properly_right_classified(
                actual_class_before, predicted_class_before, predicted_class_after
            )
            
            # Determine entropy change type
            entropy_change_type = "increase" if delta_entropy > 0 else "decrease"
            
            # Record results with JSON-safe conversions
            result = {
                "node_id": int(target_node),
                "actual_class": int(actual_class_before),
                "predicted_class_before": int(predicted_class_before),
                "predicted_class_after": int(predicted_class_after),
                "probabilities_before": convert_to_json_serializable(probs_before),
                "probabilities_after": convert_to_json_serializable(probs_after),
                "entropy_before": float(entropy_before),
                "entropy_after": float(entropy_after),
                "delta_entropy": float(delta_entropy),
                "kl_divergence": float(kl_divergence),
                "attack_history": convert_to_json_serializable(attack_history),
                "properly_misclassified": bool(properly_misclassified),
                "properly_right_classified": bool(properly_right_classified),
                "originally_correct": bool(actual_class_before == predicted_class_before),
                "after_attack_wrong": bool(actual_class_before != predicted_class_after),
                "after_attack_correct": bool(actual_class_before == predicted_class_after),
                "entropy_change_type": entropy_change_type,
                "budget_used": int(budget),
                "experiment_timestamp": float(time.time())
            }
            
            logger.info(f"Node {target_node}: ΔH={delta_entropy:.4f}, KL={kl_divergence:.4f}, "
                       f"Properly Misclassified={properly_misclassified}, "
                       f"Properly Right Classified={properly_right_classified}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing node {target_node}: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_experiment(self, target_misclassified=200, target_right_classified=200, max_nodes=1000, budget=5):
        """
        Run the experiment until we get target_misclassified properly misclassified nodes
        and target_right_classified properly right classified nodes, or reach max_nodes limit
        """
        logger.info(f"Starting Connectivity-Disruption Theory experiment")
        logger.info(f"Dataset: {self.dataset}, Surrogate: {self.surrogate_model}")
        logger.info(f"Target misclassified nodes: {target_misclassified}")
        logger.info(f"Target right classified nodes: {target_right_classified}")
        logger.info(f"Max nodes to test: {max_nodes}")
        
        nodes_tested = 0
        properly_misclassified_count = 0
        properly_right_classified_count = 0
        
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=len(self.target_nodes), 
                   desc="Finding nodes")
        
        for target_node in self.target_nodes:
            if nodes_tested >= max_nodes:
                logger.info(f"Reached maximum node limit: {max_nodes}")
                break
                
            # Check if we've reached both targets
            if (properly_misclassified_count >= target_misclassified and 
                properly_right_classified_count >= target_right_classified):
                logger.info(f"Reached both targets - Misclassified: {target_misclassified}, Right classified: {target_right_classified}")
                break
            
            result = self.run_single_node_experiment(target_node, budget)
            nodes_tested += 1
            
            if result is not None:
                self.experiment_results.append(result)
                
                # Check if properly misclassified
                if result["properly_misclassified"] and properly_misclassified_count < target_misclassified:
                    self.misclassified_results.append(result)
                    properly_misclassified_count += 1
                
                # Check if properly right classified
                if result["properly_right_classified"] and properly_right_classified_count < target_right_classified:
                    self.right_classified_results.append(result)
                    properly_right_classified_count += 1
                
                pbar.update(1)
                pbar.set_description(f"Misclassified: {properly_misclassified_count}, Right: {properly_right_classified_count}")
            
            # Save intermediate results every 100 nodes
            if nodes_tested % 100 == 0:
                self.save_results(f"intermediate_results_{nodes_tested}.json")
        
        pbar.close()
        end_time = time.time()
        
        # Compute summary statistics
        summary_misclassified = self.compute_summary_statistics_misclassified()
        summary_right_classified = self.compute_summary_statistics_right_classified()
        
        # Add experiment metadata
        experiment_metadata = {
            "total_time_seconds": float(end_time - start_time),
            "nodes_tested": int(nodes_tested),
            "properly_misclassified_count": int(properly_misclassified_count),
            "properly_right_classified_count": int(properly_right_classified_count),
            "misclassified_target_reached": bool(properly_misclassified_count >= target_misclassified),
            "right_classified_target_reached": bool(properly_right_classified_count >= target_right_classified),
            "both_targets_reached": bool(properly_misclassified_count >= target_misclassified and 
                                       properly_right_classified_count >= target_right_classified)
        }
        
        logger.info(f"Experiment completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Nodes tested: {nodes_tested}")
        logger.info(f"Properly misclassified nodes found: {properly_misclassified_count}")
        logger.info(f"Properly right classified nodes found: {properly_right_classified_count}")
        
        return {
            "experiment_metadata": experiment_metadata,
            "summary_statistics_misclassified": summary_misclassified,
            "summary_statistics_right_classified": summary_right_classified
        }
    
    def compute_summary_statistics_misclassified(self):
        """
        Compute summary statistics for properly misclassified nodes
        """
        if not self.misclassified_results:
            return {
                "dataset": self.dataset,
                "surrogate_model": self.surrogate_model,
                "total_properly_misclassified": 0,
                "message": "No properly misclassified nodes found"
            }
        
        # Extract values from properly misclassified nodes only
        delta_entropies = [r["delta_entropy"] for r in self.misclassified_results]
        kl_divergences = [r["kl_divergence"] for r in self.misclassified_results]
        
        # Separate by entropy change type
        entropy_increases = [r for r in self.misclassified_results if r["delta_entropy"] > 0]
        entropy_decreases = [r for r in self.misclassified_results if r["delta_entropy"] < 0]
        entropy_no_change = [r for r in self.misclassified_results if abs(r["delta_entropy"]) < 1e-6]
        
        # Theory validation metrics
        theorem_support_cases = len([r for r in self.misclassified_results 
                                   if r["kl_divergence"] > 0 and abs(r["delta_entropy"]) > 1e-6])
        
        summary = {
            "dataset": self.dataset,
            "surrogate_model": self.surrogate_model,
            "total_properly_misclassified": len(self.misclassified_results),
            "entropy_statistics": {
                "mean_delta_entropy": float(np.mean(delta_entropies)),
                "std_delta_entropy": float(np.std(delta_entropies)),
                "min_delta_entropy": float(np.min(delta_entropies)),
                "max_delta_entropy": float(np.max(delta_entropies)),
                "median_delta_entropy": float(np.median(delta_entropies))
            },
            "kl_divergence_statistics": {
                "mean_kl_divergence": float(np.mean(kl_divergences)),
                "std_kl_divergence": float(np.std(kl_divergences)),
                "min_kl_divergence": float(np.min(kl_divergences)),
                "max_kl_divergence": float(np.max(kl_divergences)),
                "median_kl_divergence": float(np.median(kl_divergences))
            },
            "entropy_change_analysis": {
                "entropy_increases": {
                    "count": len(entropy_increases),
                    "percentage": float(len(entropy_increases) / len(self.misclassified_results) * 100),
                    "mean_delta_entropy": float(np.mean([r["delta_entropy"] for r in entropy_increases])) if entropy_increases else 0.0,
                    "mean_kl_divergence": float(np.mean([r["kl_divergence"] for r in entropy_increases])) if entropy_increases else 0.0
                },
                "entropy_decreases": {
                    "count": len(entropy_decreases),
                    "percentage": float(len(entropy_decreases) / len(self.misclassified_results) * 100),
                    "mean_delta_entropy": float(np.mean([r["delta_entropy"] for r in entropy_decreases])) if entropy_decreases else 0.0,
                    "mean_kl_divergence": float(np.mean([r["kl_divergence"] for r in entropy_decreases])) if entropy_decreases else 0.0
                },
                "entropy_no_change": {
                    "count": len(entropy_no_change),
                    "percentage": float(len(entropy_no_change) / len(self.misclassified_results) * 100)
                }
            },
            "theorem_validation": {
                "theorem_support_cases": int(theorem_support_cases),
                "support_rate": float(theorem_support_cases / len(self.misclassified_results)),
                "kl_divergence_positive_rate": float(len([r for r in self.misclassified_results if r["kl_divergence"] > 0]) / len(self.misclassified_results)),
                "entropy_change_rate": float(len([r for r in self.misclassified_results if abs(r["delta_entropy"]) > 1e-6]) / len(self.misclassified_results))
            }
        }
        
        return summary
    
    def compute_summary_statistics_right_classified(self):
        """
        Compute summary statistics for properly right classified nodes
        """
        if not self.right_classified_results:
            return {
                "dataset": self.dataset,
                "surrogate_model": self.surrogate_model,
                "total_properly_right_classified": 0,
                "message": "No properly right classified nodes found"
            }
        
        # Extract values from properly right classified nodes only
        delta_entropies = [r["delta_entropy"] for r in self.right_classified_results]
        kl_divergences = [r["kl_divergence"] for r in self.right_classified_results]
        
        # Separate by entropy change type
        entropy_increases = [r for r in self.right_classified_results if r["delta_entropy"] > 0]
        entropy_decreases = [r for r in self.right_classified_results if r["delta_entropy"] < 0]
        entropy_no_change = [r for r in self.right_classified_results if abs(r["delta_entropy"]) < 1e-6]
        
        # Theory validation metrics for right classified nodes
        theorem_support_cases = len([r for r in self.right_classified_results 
                                   if r["kl_divergence"] > 0 and abs(r["delta_entropy"]) > 1e-6])
        
        summary = {
            "dataset": self.dataset,
            "surrogate_model": self.surrogate_model,
            "total_properly_right_classified": len(self.right_classified_results),
            "entropy_statistics": {
                "mean_delta_entropy": float(np.mean(delta_entropies)),
                "std_delta_entropy": float(np.std(delta_entropies)),
                "min_delta_entropy": float(np.min(delta_entropies)),
                "max_delta_entropy": float(np.max(delta_entropies)),
                "median_delta_entropy": float(np.median(delta_entropies))
            },
            "kl_divergence_statistics": {
                "mean_kl_divergence": float(np.mean(kl_divergences)),
                "std_kl_divergence": float(np.std(kl_divergences)),
                "min_kl_divergence": float(np.min(kl_divergences)),
                "max_kl_divergence": float(np.max(kl_divergences)),
                "median_kl_divergence": float(np.median(kl_divergences))
            },
            "entropy_change_analysis": {
                "entropy_increases": {
                    "count": len(entropy_increases),
                    "percentage": float(len(entropy_increases) / len(self.right_classified_results) * 100),
                    "mean_delta_entropy": float(np.mean([r["delta_entropy"] for r in entropy_increases])) if entropy_increases else 0.0,
                    "mean_kl_divergence": float(np.mean([r["kl_divergence"] for r in entropy_increases])) if entropy_increases else 0.0
                },
                "entropy_decreases": {
                    "count": len(entropy_decreases),
                    "percentage": float(len(entropy_decreases) / len(self.right_classified_results) * 100),
                    "mean_delta_entropy": float(np.mean([r["delta_entropy"] for r in entropy_decreases])) if entropy_decreases else 0.0,
                    "mean_kl_divergence": float(np.mean([r["kl_divergence"] for r in entropy_decreases])) if entropy_decreases else 0.0
                },
                "entropy_no_change": {
                    "count": len(entropy_no_change),
                    "percentage": float(len(entropy_no_change) / len(self.right_classified_results) * 100)
                }
            },
            "theorem_validation": {
                "theorem_support_cases": int(theorem_support_cases),
                "support_rate": float(theorem_support_cases / len(self.right_classified_results)),
                "kl_divergence_positive_rate": float(len([r for r in self.right_classified_results if r["kl_divergence"] > 0]) / len(self.right_classified_results)),
                "entropy_change_rate": float(len([r for r in self.right_classified_results if abs(r["delta_entropy"]) > 1e-6]) / len(self.right_classified_results))
            }
        }
        
        return summary
    
    def save_results(self, filename="connectivity_disruption_results.json"):
        """
        Save experiment results to JSON file, including both misclassified and right classified nodes
        """
        summary_stats_misclassified = self.compute_summary_statistics_misclassified()
        summary_stats_right_classified = self.compute_summary_statistics_right_classified()
        
        output_data = {
            "experiment_metadata": {
                "dataset": self.dataset,
                "surrogate_model": self.surrogate_model,
                "defense_model": self.defense_model,
                "timestamp": float(time.time()),
                "experiment_type": "connectivity_disruption_theory_validation"
            },
            "summary_statistics_misclassified": summary_stats_misclassified,
            "summary_statistics_right_classified": summary_stats_right_classified,
            "properly_misclassified_results": convert_to_json_serializable(self.misclassified_results),
            "properly_right_classified_results": convert_to_json_serializable(self.right_classified_results),
            "all_tested_results_count": len(self.experiment_results),
            "theory_proof_data": {
                "total_properly_misclassified": len(self.misclassified_results),
                "total_properly_right_classified": len(self.right_classified_results),
                "misclassified_entropy_increases": len([r for r in self.misclassified_results if r["delta_entropy"] > 0]),
                "misclassified_entropy_decreases": len([r for r in self.misclassified_results if r["delta_entropy"] < 0]),
                "misclassified_positive_kl_divergences": len([r for r in self.misclassified_results if r["kl_divergence"] > 0]),
                "right_classified_entropy_increases": len([r for r in self.right_classified_results if r["delta_entropy"] > 0]),
                "right_classified_entropy_decreases": len([r for r in self.right_classified_results if r["delta_entropy"] < 0]),
                "right_classified_positive_kl_divergences": len([r for r in self.right_classified_results if r["kl_divergence"] > 0]),
                "misclassified_theorem_support_evidence": summary_stats_misclassified.get("theorem_validation", {}),
                "right_classified_theorem_support_evidence": summary_stats_right_classified.get("theorem_validation", {})
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Results saved to {filename}")
        logger.info(f"Saved {len(self.misclassified_results)} properly misclassified nodes")
        logger.info(f"Saved {len(self.right_classified_results)} properly right classified nodes")
    
    def analyze_theorem_validation(self):
        """
        Analyze results to validate the Connectivity-Disruption Theory for both groups
        """
        results = {}
        
        # Analyze misclassified nodes
        if self.misclassified_results:
            theorem_support_cases_mis = 0
            kl_positive_cases_mis = 0
            entropy_change_cases_mis = 0
            
            for result in self.misclassified_results:
                delta_h = result["delta_entropy"]
                kl_div = result["kl_divergence"]
                
                if kl_div > 0:
                    kl_positive_cases_mis += 1
                
                if abs(delta_h) > 1e-6:
                    entropy_change_cases_mis += 1
                
                if kl_div > 0 and abs(delta_h) > 1e-6:
                    theorem_support_cases_mis += 1
            
            total_misclassified = len(self.misclassified_results)
            results["misclassified"] = {
                "total_properly_misclassified": total_misclassified,
                "kl_positive_cases": kl_positive_cases_mis,
                "entropy_change_cases": entropy_change_cases_mis,
                "theorem_support_cases": theorem_support_cases_mis,
                "support_rate": theorem_support_cases_mis / total_misclassified if total_misclassified > 0 else 0,
                "kl_positive_rate": kl_positive_cases_mis / total_misclassified if total_misclassified > 0 else 0,
                "entropy_change_rate": entropy_change_cases_mis / total_misclassified if total_misclassified > 0 else 0
            }
        
        # Analyze right classified nodes
        if self.right_classified_results:
            theorem_support_cases_right = 0
            kl_positive_cases_right = 0
            entropy_change_cases_right = 0
            
            for result in self.right_classified_results:
                delta_h = result["delta_entropy"]
                kl_div = result["kl_divergence"]
                
                if kl_div > 0:
                    kl_positive_cases_right += 1
                
                if abs(delta_h) > 1e-6:
                    entropy_change_cases_right += 1
                
                if kl_div > 0 and abs(delta_h) > 1e-6:
                    theorem_support_cases_right += 1
            
            total_right_classified = len(self.right_classified_results)
            results["right_classified"] = {
                "total_properly_right_classified": total_right_classified,
                "kl_positive_cases": kl_positive_cases_right,
                "entropy_change_cases": entropy_change_cases_right,
                "theorem_support_cases": theorem_support_cases_right,
                "support_rate": theorem_support_cases_right / total_right_classified if total_right_classified > 0 else 0,
                "kl_positive_rate": kl_positive_cases_right / total_right_classified if total_right_classified > 0 else 0,
                "entropy_change_rate": entropy_change_cases_right / total_right_classified if total_right_classified > 0 else 0
            }
        
        # Print analysis
        logger.info("=== Connectivity-Disruption Theory Validation Analysis ===")
        
        if "misclassified" in results:
            mis_results = results["misclassified"]
            logger.info(f"MISCLASSIFIED NODES:")
            logger.info(f"  Total properly misclassified nodes: {mis_results['total_properly_misclassified']}")
            logger.info(f"  Cases with positive KL divergence: {mis_results['kl_positive_cases']} ({mis_results['kl_positive_rate']:.2%})")
            logger.info(f"  Cases with entropy change: {mis_results['entropy_change_cases']} ({mis_results['entropy_change_rate']:.2%})")
            logger.info(f"  Cases supporting theorem: {mis_results['theorem_support_cases']}")
            logger.info(f"  Theorem support rate: {mis_results['support_rate']:.2%}")
        
        if "right_classified" in results:
            right_results = results["right_classified"]
            logger.info(f"RIGHT CLASSIFIED NODES:")
            logger.info(f"  Total properly right classified nodes: {right_results['total_properly_right_classified']}")
            logger.info(f"  Cases with positive KL divergence: {right_results['kl_positive_cases']} ({right_results['kl_positive_rate']:.2%})")
            logger.info(f"  Cases with entropy change: {right_results['entropy_change_cases']} ({right_results['entropy_change_rate']:.2%})")
            logger.info(f"  Cases supporting theorem: {right_results['theorem_support_cases']}")
            logger.info(f"  Theorem support rate: {right_results['support_rate']:.2%}")
        
        return results

def main():
    """
    Main function to run the Connectivity-Disruption Theory experiment
    """
    # Experiment configuration
    DATASET = 'cora'
    SURROGATE_MODEL = 'gcn'
    DEFENSE_MODEL = 'gcn'
    TARGET_MISCLASSIFIED = 200
    TARGET_RIGHT_CLASSIFIED = 200
    MAX_NODES = 1000
    BUDGET = 5
    
    # Initialize experiment
    experiment = ConnectivityDisruptionExperiment(
        surrogate_model=SURROGATE_MODEL,
        dataset=DATASET,
        defense_model=DEFENSE_MODEL
    )
    
    # Run experiment
    logger.info("Starting Connectivity-Disruption Theory Experimental Proof")
    summary = experiment.run_experiment(
        target_misclassified=TARGET_MISCLASSIFIED,
        target_right_classified=TARGET_RIGHT_CLASSIFIED,
        max_nodes=MAX_NODES, 
        budget=BUDGET
    )
    
    # Save results
    experiment.save_results("connectivity_disruption_experiment_results.json")
    
    # Analyze theorem validation
    validation_results = experiment.analyze_theorem_validation()
    
    # Print final summary
    print("\n" + "="*70)
    print("CONNECTIVITY-DISRUPTION THEORY EXPERIMENTAL PROOF RESULTS")
    print("="*70)
    print(f"Dataset: {DATASET}")
    print(f"Nodes tested: {summary['experiment_metadata']['nodes_tested']}")
    print(f"Properly misclassified nodes found: {summary['experiment_metadata']['properly_misclassified_count']}")
    print(f"Properly right classified nodes found: {summary['experiment_metadata']['properly_right_classified_count']}")
    print(f"Misclassified target reached: {summary['experiment_metadata']['misclassified_target_reached']}")
    print(f"Right classified target reached: {summary['experiment_metadata']['right_classified_target_reached']}")
    print(f"Both targets reached: {summary['experiment_metadata']['both_targets_reached']}")
    
    # Print misclassified nodes statistics
    if validation_results.get("misclassified"):
        mis_val = validation_results["misclassified"]
        mis_stats = summary["summary_statistics_misclassified"]
        print(f"\nMISCLASSIFIED NODES ANALYSIS:")
        print(f"  Theorem support rate: {mis_val['support_rate']:.2%}")
        print(f"  KL divergence positive rate: {mis_val['kl_positive_rate']:.2%}")
        print(f"  Entropy change rate: {mis_val['entropy_change_rate']:.2%}")
        if 'entropy_statistics' in mis_stats:
            print(f"  Mean ΔH: {mis_stats['entropy_statistics']['mean_delta_entropy']:.4f}")
            print(f"  Mean KL divergence: {mis_stats['kl_divergence_statistics']['mean_kl_divergence']:.4f}")
    
    # Print right classified nodes statistics
    if validation_results.get("right_classified"):
        right_val = validation_results["right_classified"]
        right_stats = summary["summary_statistics_right_classified"]
        print(f"\nRIGHT CLASSIFIED NODES ANALYSIS:")
        print(f"  Theorem support rate: {right_val['support_rate']:.2%}")
        print(f"  KL divergence positive rate: {right_val['kl_positive_rate']:.2%}")
        print(f"  Entropy change rate: {right_val['entropy_change_rate']:.2%}")
        if 'entropy_statistics' in right_stats:
            print(f"  Mean ΔH: {right_stats['entropy_statistics']['mean_delta_entropy']:.4f}")
            print(f"  Mean KL divergence: {right_stats['kl_divergence_statistics']['mean_kl_divergence']:.4f}")
    
    print("="*70)
    
    return experiment

if __name__ == "__main__":
    experiment = main()