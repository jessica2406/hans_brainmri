class SymbolicReasoner:
    def __init__(self):
        # We now define unique features for every class in your Kaggle dataset
        self.rules = {
            "empyema": ["crescentic_fluid", "restricted_diffusion", "peripheral_enhancement"],
            "meningioma": ["well_defined_border", "dural_tail", "solid_mass"],
            "glioma": ["irregular_border", "infiltrative_growth", "edema"],
            "pituitary": ["central_location", "sellar_mass"],
            "no_tumor": ["brain_symmetry", "clear_sulci"]
        }

    def reason(self, neural_features):
        logic_results = {}
        proof_tree = []

        for diagnosis, symptoms in self.rules.items():
            # Calculate how many symptoms the AI 'sees'
            # We take the average probability of the symptoms for that diagnosis
            score = sum([neural_features.get(s, 0) for s in symptoms]) / len(symptoms)
            logic_results[diagnosis] = score
            
            if score > 0.6:
                proof_tree.append(f"✅ CONFIRMED: {diagnosis.upper()} (Strong evidence of {', '.join(symptoms)})")
            elif score > 0.2:
                proof_tree.append(f"⚠️ PROVISIONAL: Possible {diagnosis.upper()} detected ({score:.1%})")
            else:
                proof_tree.append(f"❌ REJECTED: {diagnosis.upper()} (Insufficient evidence)")
        
        # The final decision is the one with the highest logical evidence
        final_decision = max(logic_results, key=logic_results.get)
        return final_decision, proof_tree