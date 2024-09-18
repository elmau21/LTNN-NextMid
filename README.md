# LTNN: Logic Meets Attention

**Author**: Mauricio Pacheco Lizama  
**Email**: mauriciopacheco.nextmid@gmail.com  
**Repository**: [LTNN GitHub Repo](https://github.com/elmau21/LTNN)

## Overview

**LTNN (Logical Transformer Neural Network)** is a novel architecture that combines the structured reasoning of Logical Neural Networks (LNNs) with the powerful contextual learning abilities of transformers. This model is designed to enhance explainability, context-awareness, and logical consistency in Natural Language Processing (NLP) tasks by integrating logical reasoning into the attention mechanisms of transformers.

### Key Features:
- **Logical Masking in Attention**: Embeds logical constraints into the transformer’s attention mechanism to ensure attention scores adhere to logical rules.
- **Bidirectional Reasoning**: Like LNNs, LTNN supports forward and backward inference for improved reasoning under uncertainty.
- **Explainability**: Provides logical explanations for model decisions, addressing one of the key limitations of standard transformers.

## Architecture

The LTNN architecture extends transformers by introducing logical operations into the attention mechanism:
- Traditional attention weights are modified using logical masks, guiding the attention process based on both learned relationships and predefined logical rules.
- Real-valued logic operations enable handling ambiguity and uncertainty, ensuring logical consistency across complex relationships.

## Mathematical Foundations

LTNNs use real-valued logic, where neurons perform logical operations with truth values ranging from 0 to 1. This allows LTNNs to:
- **Handle Uncertainty**: By employing real-valued logical AND and OR operations.
- **Incorporate Logical Constraints**: Via weighted logic combined with neural network learning, where the logical loss penalizes violations of predefined rules.

For more mathematical details, refer to the (Available soon).

## Applications

LTNNs excel in several NLP tasks, including:
- **Ambiguity Resolution**: Handling ambiguous meanings in language using context-sensitive logical rules.
- **Knowledge-Based Question Answering**: Providing interpretable answers by reasoning over structured knowledge.
- **Coreference Resolution**: Enforcing logical consistency between related entities in texts, improving performance in tasks like summarization and translation.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Additional dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/elmau21/LTNN.git
   cd LTNN
