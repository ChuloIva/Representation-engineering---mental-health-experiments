# 🧠 Cognitive Concept Pairs

This directory contains the core experimental data: cognitive concept pairs used for representation engineering.

## 📄 Data Files

### `positive_concepts.csv`
**Positive psychological concepts and therapeutic targets**
- Concepts representing psychological well-being
- Used as positive targets in contrastive vector extraction
- Categories: intrinsic motivation, social connection, personal growth

### `negative_concepts.csv`
**Negative psychological concepts and depressive states**
- Concepts representing psychological distress
- Used as negative contrasts in vector extraction
- Categories: isolation, helplessness, stagnation

### `therapeutic_techniques.csv`
**Evidence-based therapeutic interventions and techniques**
- Established therapeutic approaches and concepts
- Used for validation against known effective interventions
- Categories: CBT, mindfulness, behavioral activation

## 🎯 Experimental Framework

### Contrastive Pairs
Each concept pair represents a psychological dimension:

```
Negative Concept ←→ Positive Concept
    (avoid)           (approach)
```

**Examples:**
- `Isolation` ←→ `Community`
- `Helplessness` ←→ `Empowerment`
- `Rumination` ←→ `Mindfulness`

### Vector Extraction
1. **Prompt Generation**: Create context-specific prompts for each concept
2. **Activation Recording**: Capture model activations during generation
3. **Contrastive Analysis**: Extract directional vectors between positive/negative pairs
4. **Normalization**: Standardize vector magnitudes for consistent application

### Categories

#### Intrinsic Motivation (Basic Psychological Needs)
- **Autonomy**: Self-direction vs. External control
- **Competence**: Mastery vs. Incompetence
- **Relatedness**: Connection vs. Isolation

#### Extrinsic Motivation (Socially Constructed Goals)
- **Recognition**: Fame vs. Obscurity
- **Appearance**: Beauty vs. Neglect
- **Material**: Wealth vs. Poverty

#### Therapeutic Approaches
- **Cognitive**: Adaptive thinking vs. Cognitive distortions
- **Behavioral**: Activation vs. Avoidance
- **Emotional**: Regulation vs. Dysregulation

## 📊 File Format

```csv
concept,category,valence,description,examples,therapeutic_target
Community,intrinsic,positive,"Social connection and belonging","friendship, support groups, community involvement",true
Isolation,intrinsic,negative,"Social disconnection and loneliness","alone, excluded, withdrawn",false
```

**Columns:**
- `concept`: The psychological concept name
- `category`: Conceptual category (intrinsic/extrinsic/therapeutic)
- `valence`: Positive or negative valence
- `description`: Detailed concept description
- `examples`: Concrete examples and manifestations
- `therapeutic_target`: Whether this is a known therapeutic target

## 🔬 Research Foundation

These concepts are grounded in:
- **Self-Determination Theory**: Intrinsic vs. extrinsic motivation research
- **Clinical Psychology**: Evidence-based therapeutic targets
- **Positive Psychology**: Well-being and flourishing research
- **Cognitive Science**: Conceptual representation research

## 🔗 Usage

These files are used by:
- **Vector extraction scripts**: Generate representation engineering vectors
- **Evaluation notebooks**: Analyze concept effectiveness
- **Visualization tools**: Create concept relationship maps
- **Statistical analysis**: Test psychological hypotheses