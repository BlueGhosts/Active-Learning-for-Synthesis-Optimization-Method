# ALSO: Active Learning for Synthesis Optimization

ALSO (Active Learning for Synthesis Optimization) is a Python-based active learning framework specifically designed for the directed synthesis of zeolite materials. It enables rapid identification of optimal synthesis conditions with minimal experimental trials by efficiently exploring multi-dimensional chemical reaction spaces, significantly reducing the cost of exploring complex chemical spaces.

<img width="1344" height="944" alt="图片 1" src="https://github.com/user-attachments/assets/607839d8-c46b-4419-8340-052361c55ae4" />

## Core Features
- **Chemical Space Management**: Supports multi-format data reading (Excel/CSV), experimental parameter range definition, data normalization, and full-lifecycle data management.
- **Multi-Model Fusion Prediction**: Integrates SVM (phase classification), XGBoost (feature regression), and a custom gradient algorithm (trend capture) to achieve accurate target feature scoring.
- **Stagewise Uncertainty Assessment**: Uses distance-based assessment for scarce data and model disagreement-based assessment for sufficient data to accelerate model convergence.
- **Intelligent Experiment Recommendation**: Synthesizes performance scores and uncertainty to differentially recommend experimental conditions in synthesizable/non-synthesizable spaces, balancing exploration and exploitation.
- **Flexible Parameter Configuration**: Customizes core parameters via configuration files, supporting visual output of results and multi-round iterative workflows.

## Methodology
The ALSO framework follows a cyclic iterative logic of "Prediction - Validation - Model Optimization", with three key design components:
1. **Phase Prediction**: Identifies regions in the chemical space where target zeolites can be synthesized using an SVM classification algorithm.
2. **Target Feature Scoring**: Combines XGBoost (local accuracy) and a custom gradient algorithm (global trend capture) to predict performance metrics.
3. **Uncertainty Assessment**: Quantifies spatial uncertainty in stages, prioritizing experimental conditions with high exploration value for recommendation.
4. **Experiment Recommendation**: Differentially recommends conditions within synthesizable spaces (high score + high uncertainty) and outside synthesizable spaces (potential phases).


## Code Structure
```plaintext
ALSO/
├── also/
│   ├── __init__.py                # Package initialization
│   ├── ExperimentSpace.py         # Chemical space module: data reading, feature engineering, normalization
│   ├── MLPredicition.py           # ML prediction module: encapsulation of SVM/XGBoost models
│   ├── ExperimentPlan.py          # Experiment planning module: uncertainty assessment, condition ranking
│   ├── Gradient.py                # Gradient prediction module: custom gradient regression algorithm
│   ├── Output.py                  # Output module: result visualization and file export
│   ├── Sampling.py                # Sampling module: candidate point generation in chemical space
│   ├── ReadParameter.py           # Parameter reading module: parse parameter.ini
│   ├── parameter.ini              # Parameter configuration file: experiment/model/output settings
│   └── main.py                    # Main workflow module: iterative logic, module orchestration
└── README.md                      # Documentation
```

### Core Module Explanation
| Module File            | Core Function                                                                 |
|------------------------|-------------------------------------------------------------------------------|
| ExperimentSpace.py     | Manages the full lifecycle of experimental data and constructs chemical spaces |
| MLPredicition.py       | Encapsulates SVM (phase classification) and XGBoost (feature regression) with unified training/prediction interfaces |
| ExperimentPlan.py      | Calculates uncertainty scores for candidate points and recommends high-value experimental conditions |
| Gradient.py            | Implements custom gradient regression algorithm to capture smooth trends in performance space |
| ReadParameter.py       | Parses parameter.ini to load experiment/model/output configurations            |
| main.py                | Orchestrates all modules to implement the iterative closed loop of "Prediction - Validation - Update" |

## Application Scenarios
The ALSO framework is suitable for end-to-end optimization of zeolite synthesis, with validated core scenarios:
- Directed optimization of high-performance TS-1 oxidative desulfurization catalysts
- Improvement of ZEO-2 zeolite synthesis stability
- Rapid mapping of phase diagrams for multi-phase Si/Al systems
- Screening of organic structure-directing agents (OSDAs) with high binding energy

