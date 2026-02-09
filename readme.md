# Unboxing the Black Box: The Lazy Artist -- Computer Vision Task

**Author:** P. Srivatsav Reddy  
**Roll Number:** 2025121016  
**Submitted to:** PreCog Research Group, IIIT Hyderabad

---

## Overview

This project investigates how Convolutional Neural Networks exploit spurious correlations (shortcut learning) and what can be done about it. Using a synthetically biased Colored-MNIST dataset, we train models that learn to associate background colors with digit classes instead of learning actual digit shapes. We then diagnose this failure using interpretability tools, intervene to fix it, attack the models adversarially, and decompose their internal representations using Sparse Autoencoders.

The work spans six tasks -- from dataset creation to mechanistic interpretability -- and documents every experiment, including the ones that failed. All results are logged inside the notebooks themselves.

---

## Directory Structure

```
precogcvtask-unboxingblackbox-CNN-s/
|
|-- readme.md                 --> This file. Project documentation.
|-- task0.ipynb               --> Task 0: Biased Colored-MNIST dataset generation
|-- task1simplecnn.ipynb      --> Task 1: Training and evaluating a simple 3-layer CNN (LazyCNN)
|-- task1resnet18.ipynb        --> Task 1: Training and evaluating a fine-tuned ResNet-18
|-- task2final.ipynb           --> Task 2: Neuron probing, feature visualization, deep dreaming, embeddings
|-- task3final.ipynb           --> Task 3: Grad-CAM implementation from scratch + library comparison
|-- task4final.ipynb           --> Task 4: Intervention -- BatchNorm and color gradient penalty training
|-- task5final.ipynb           --> Task 5: Targeted adversarial attacks (FGSM, PGD, DeepFool)
|-- task6final.ipynb           --> Task 6: Sparse Autoencoder decomposition of hidden states
```

**External Dataset (Kaggle):** [Colored-MNIST with Spurious Correlation](https://www.kaggle.com/datasets/poreddysr/c2ty2bcoulouredspurious)

All training was performed on Kaggle with GPU T4 runtime. Model checkpoints were saved on Kaggle and loaded across notebooks as Kaggle dataset inputs.

---

## Dependencies

- Python 3.10+
- PyTorch (torch, torchvision)
- NumPy
- Matplotlib
- Seaborn
- Pillow (PIL)
- scikit-learn (StratifiedShuffleSplit, confusion_matrix)
- tqdm
- ipywidgets (for interactive deep dream slider in Task 2)
- pytorch-grad-cam (installed from GitHub in Task 3, used only for validation against scratch implementation)

Install all dependencies:

```bash
pip install torch torchvision numpy matplotlib seaborn pillow scikit-learn tqdm ipywidgets
```

For Task 3 (library Grad-CAM comparison only):

```bash
pip install "git+https://github.com/jacobgil/pytorch-grad-cam.git"
```

---

## How to Run

1. **Dataset:** Download the Colored-MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/poreddysr/c2ty2bcoulouredspurious), or regenerate it by running `task0.ipynb` on standard MNIST.
2. **Training:** Run `task1simplecnn.ipynb` and `task1resnet18.ipynb` on Kaggle (GPU T4 recommended) to train the biased models.
3. **Subsequent tasks:** Each notebook from Task 2 onward loads pre-trained checkpoints. Update the checkpoint paths at the top of each notebook to point to your saved model files.
4. **Execution order:** Tasks are meant to be run sequentially (0 through 6), since later tasks depend on models and data produced by earlier ones.

All notebooks are self-contained. Every cell can be run top-to-bottom -- all results, plots, and metrics are logged inline.

---

## Task-by-Task Summary

### Task 0 -- The Biased Canvas

**Notebook:** `task0.ipynb`

**Goal:** Generate a Colored-MNIST dataset with a deliberate spurious correlation between digit identity and background color.

**What was done:**

- Each digit (0-9) is assigned a dominant background color: 0=Red, 1=Green, 2=Blue, 3=Yellow, 4=Purple, 5=Orange, 6=Cyan, 7=Magenta, 8=Brown, 9=Pink.
- **Easy (Train) set:** 95% of each digit's images use the assigned background color. The remaining 5% use a random different color. Built from the 60,000 MNIST training images.
- **Hard (Test) set:** The color-digit correlation is broken. Each digit is given a color that does *not* match its assigned one. Built from the 10,000 MNIST test images.
- The background is textured (not flat) -- Gaussian noise (+/-20 pixel values) is added to the base color to create a noisy, non-trivial texture.
- The foreground digit strokes are rendered in gray (inverted grayscale, clamped to 180-255 range).
- Images are saved as 28x28 RGB PNGs organized in `colored_mnist/type2_background/{easy_train,hard_test}/{0..9}/` directory structure.
- A Kaggle data card was created and published.

**Key output:** 60,000 easy-train images + 10,000 hard-test images, all with textured colored backgrounds and gray foreground strokes.

---

### Task 1 -- The Cheater

**Notebooks:** `task1simplecnn.ipynb`, `task1resnet18.ipynb`

**Goal:** Train standard CNNs on the biased dataset and demonstrate that they learn the color shortcut instead of digit shapes.

**Data handling:**

- Used a 25% stratified subset of the full dataset for faster iteration (approximately 12,000 train / 3,000 val / 2,500 hard-test).
- Proper stratified splitting (via `StratifiedShuffleSplit`) was applied to preserve class balance across train, validation, and test sets.
- DataLoaders used `num_workers=2` and `pin_memory=True` for efficient GPU loading.

**Model 1 -- LazyCNN (Simple 3-Layer CNN):**

- Architecture: Conv2d(3,6,9) -> Conv2d(6,8,7) -> Conv2d(8,16,3) -> FC(784,10). No BatchNorm, no pooling layers.
- Optimizer: SGD, lr=0.01. Loss: CrossEntropyLoss. Trained for 15 epochs.
- **Training accuracy: 95.15%**
- **Easy validation accuracy: 95.13%**
- **Hard test accuracy: 7.04%**

**Model 2 -- ResNet-18 (Transfer Learning):**

- Pretrained ResNet-18 weights loaded. Input resized to 224x224 with ImageNet normalization.
- Final FC layer replaced for 10-class output.
- All layers frozen except `layer4` and `fc` (fine-tuning the final block only).
- Optimizer: SGD, lr=0.01. Trained for 15 epochs.
- **Training accuracy: 95.36%**
- **Easy validation accuracy: 95.23%**
- **Hard test accuracy: 7.56%**

**Analysis performed:**

- Confusion matrices for both easy validation and hard test sets.
- Color-vs-prediction heatmap: a matrix showing how background color drives predictions on the hard test set. For example, images with Red backgrounds are overwhelmingly predicted as 0 regardless of the actual digit.
- Specific proof of color bias: a Red digit "1" from the hard test set being confidently predicted as "0", demonstrating the model has learned "Red = 0" rather than the shape of each digit.

**Key finding:** Both models achieve above 95% on the easy set but collapse to near-random on the hard set. The confusion matrices and color-prediction heatmaps prove the models rely on background color, not digit shape. Even ResNet-18 -- a much larger architecture -- is equally biased, confirming that model capacity alone does not prevent shortcut learning.

---

### Task 2 -- The Prober

**Notebook:** `task2final.ipynb`

**Goal:** Visualize what the neurons in the trained LazyCNN and ResNet-18 are actually "seeing."

**What was done:**

1. **Filter weight visualization (Conv1):** Plotted all 6 conv1 filters across 3 input channels using a symmetric seismic colormap. Most filter weights are focused on capturing background color patterns rather than edge-like digit features.

2. **Feature map visualization:** Passed a sample image through the LazyCNN and plotted feature maps at every layer (Conv1, ReLU1, Conv2, ReLU2, Conv3, ReLU3) with proper global min-max scaling per layer. The activations are visibly color-dominated and texture-focused.

3. **Image embedding inversion:** Starting from random noise, optimized an input image to match the internal representation (embedding) of a reference image at each layer. The reconstructions from deeper layers preserve color and texture information but lose digit shape -- confirming the model stores color more faithfully than shape.

4. **Deep dreaming:** Implemented gradient ascent from actual images to maximize target class activations. Created an interactive slider to observe how dream patterns evolve over iterations. The amplified features are overwhelmingly color- and texture-based.

5. **Neuron-level activation maximization (LazyCNN):** For each neuron in conv1, conv2, and conv3, optimized an image from shared random noise using three modes:
   - Raw activation sum
   - Channel-wise mean activation
   - Spatial activation (center neuron)
   
   The resulting preferred images for conv3 show strong color and texture patterns with minimal digit-like structure, confirming the neurons have learned color rather than shape features.

6. **Neuron-level probing (ResNet-18):** Applied the same gradient ascent procedure to ResNet-18 neurons (layer2, layer3). The resulting visualizations show textured, polysemantic activations -- neurons respond to mixtures of color patches and vague structural patterns, confirming polysemanticity in the biased model.

**Key finding:** Across both models, neurons at every depth preferentially encode background color and texture. Shape-related features are either absent or weak. The activations are "colored and texturedly activated" -- the models have genuinely internalized the spurious correlation at the representation level.

---

### Task 3 -- The Interrogation

**Notebook:** `task3final.ipynb`

**Goal:** Implement Grad-CAM from scratch and use it to prove mathematically where the model is looking.

**Implementation:**

- `GradCAMScratch` class: hooks into a target convolutional layer, captures forward activations and backward gradients, computes channel-wise importance weights (global average pooling of gradients), produces a weighted sum of activation maps followed by ReLU, then upsamples to input resolution via bilinear interpolation.
- The implementation follows the original Selvaraju et al. (2017) paper.
- Results are validated against the `pytorch-grad-cam` library. The coarse heatmaps are compared with Mean Absolute Error (MAE), and both scratch and library versions are displayed side-by-side.

**Experiments:**

- **LazyCNN on biased image (Red 0):** The Grad-CAM heatmap smears across the colored background rather than focusing on the zero's shape. The model's attention is on the red pixels.
- **LazyCNN on conflicting image (Green 0):** The heatmap shifts and becomes confused -- it tries to look at the green background but finds no evidence for the predicted class there.
- **ResNet-18 on biased image (Red 0):** Similar pattern -- attention is on the background color region.
- **ResNet-18 on conflicting image (Green 0):** Attention becomes diffuse and uncertain.

**Note:** There is a known interpolation artifact when applying Grad-CAM to ResNet-18 due to the resolution mismatch between the 28x28 Colored-MNIST images and ResNet-18's expected 224x224 input.

**Key finding:** The Grad-CAM visualizations provide concrete visual proof that both models attend to background color regions rather than the digit stroke regions. This is the "smoking gun" for shortcut learning.

---

### Task 4 -- The Intervention

**Notebook:** `task4final.ipynb`

**Goal:** Fix the shortcut learning problem without modifying the dataset (still trained on the 95%-biased easy set).

**Architecture -- RobustCNN:**

- Conv2d(3,32,5) + BatchNorm2d + ReLU + MaxPool2d -> Conv2d(32,64,3) + BatchNorm2d + ReLU + MaxPool2d -> Conv2d(64,128,3) + BatchNorm2d + ReLU -> FC(128*7*7, 10)
- Key architectural change: **BatchNorm2d** after each convolutional layer, plus MaxPool2d for spatial downsampling.

**Method 1 -- BatchNorm only:**

- Hypothesis: BatchNorm provides translational invariance by normalizing feature activations, which could reduce the model's ability to rely on global color statistics.
- Optimizer: Adam, lr=1e-3. Trained for 20 epochs with early stopping.
- **Training accuracy: 99.38%**
- **Validation accuracy: 98.27%**
- **Hard test accuracy: 80.84%**

**Method 2 -- BatchNorm + Color Gradient Penalty:**

- Hypothesis: Explicitly penalizing the model for relying on color channels should force it to learn color-invariant features.
- Implemented a custom `color_gradient_penalty` function: computes the gradient of the predicted class score with respect to the input image, then measures the variance across the R, G, B gradient channels. High variance means the model treats color channels differently (relying on color), so this variance is added as a penalty term.
- Used a custom lambda scheduler (linearly increasing from 0 to 10 over training) -- analogous to a learning rate scheduler but for the penalty weight. Start with no penalty early on so the model can learn basic features, then gradually enforce color invariance.
- Optimizer: Adam, lr=1e-3. Trained for 20 epochs with early stopping.
- **Training accuracy: 99.79%**
- **Validation accuracy: 98.93%**
- **Hard test accuracy: 83.76%**

**Analysis:** Confusion matrices and color-vs-prediction heatmaps confirm the robust model no longer maps colors to digits. The predictions on the hard test set are distributed based on actual digit identity, not background color.

**Key finding:** Adding BatchNorm alone provides a significant boost (7% to 81%). Adding the color gradient penalty on top pushes it further to 84%. The model goes from completely fooled to respectably accurate on the adversarial test set, all while training on the same 95%-biased dataset.

---

### Task 5 -- The Invisible Cloak

**Notebook:** `task5final.ipynb`

**Goal:** Perform targeted adversarial attacks on both the lazy model and the robust model. Make images of "7" be classified as "3" with >90% confidence, while keeping the perturbation invisible (L-infinity < 0.05).

**Image selection:** Found the top-5 images of digit "7" where both models predict 7 and the second-highest logit is 3 (the closest natural confusion pair).

**Three attack methods implemented:**

1. **Targeted FGSM (Fast Gradient Sign Method):** Single-step attack. Takes one gradient step in the direction that maximizes the target class score. Simple but often insufficient for high-confidence targeted attacks.

2. **Targeted PGD (Projected Gradient Descent):** Iterative version of FGSM. Takes small alpha-sized steps (alpha = eps/10), projects back to the L-infinity ball after each step, runs up to 200 iterations. This is the standard strong white-box attack.

3. **Enhanced Targeted DeepFool:** Iterative linearization-based attack. Computes the minimal perturbation to cross the decision boundary toward the target class using the difference of gradients between the target and predicted class. Projects to L-infinity ball at each step. Up to 500 iterations.

**Constraint enforcement:** All three attacks strictly enforce L-infinity <= 0.05 and clamp pixels to [0, 1] at every step.

**Results:**

- The lazy model (LazyCNN) was attacked successfully on multiple images. Because it relies on color shortcuts rather than robust shape features, targeted pixel-level perturbations can effectively redirect its predictions.
- The robust model (RobustCNN from Task 4) was also attacked, but the comparison reveals differences in required perturbation magnitude and success rates.
- L-infinity values achieved were in the range of 0.04 (just under the 0.05 threshold).

**Failed experiment (documented):** Initially attempted to use OPTUNA (Bayesian hyperparameter optimization library) to find a universal adversarial perturbation -- a single noise pattern that would make every "7" predict as "3." The idea was to start from the difference between the most-confident-3 image and least-confident-7 image, then optimize with OPTUNA. This failed because the lazy model's per-pixel color dependence makes a universal perturbation infeasible -- each image has a different background color, so no single perturbation generalizes.

**Key finding:** Both models can be fooled, but for different reasons. The lazy model is vulnerable because its decision logic is based on global color statistics that are easy to perturb. The robust model requires more targeted, shape-aware perturbations. The comparison quantifies the difference in attack difficulty between shortcut-reliant and properly-trained models.

---

### Task 6 -- The Decomposition

**Notebook:** `task6final.ipynb`

**Goal:** Use Sparse Autoencoders (SAEs) to decompose the hidden representations of the lazy model into interpretable features and identify whether color features are explicitly encoded.

**Architecture -- Sparse Autoencoder:**

- Input: flattened conv3 output from LazyCNN, dimension 16*7*7 = 784. Also trained a separate SAE on raw pixel space (3*28*28 = 2352 -> 128 -> 2352).
- Hidden dimension: 128 bottleneck neurons.
- Activation: ReLU on encoder output, Sigmoid on decoder output.
- Loss: MSE reconstruction loss + lambda * KL-divergence sparsity penalty (target sparsity rho = 0.05, lambda = 1e-3).
- Gradient clipping (max_norm=1.0) for training stability.
- Xavier initialization on weights, zeros on biases.
- Trained for 50 epochs with Adam optimizer (lr=1e-3), batch size 256.

**What was done:**

1. **Hidden state extraction:** Passed all easy-subset images through LazyCNN's conv layers, extracted the flattened conv3 outputs as hidden state vectors.

2. **SAE training:** Trained the autoencoder to reconstruct the input (pixel-space SAE) with a sparsity constraint, forcing the 128 bottleneck neurons to develop specialized, interpretable features.

3. **Reconstruction quality:** Plotted original vs. reconstructed images. The SAE captures the dominant color and rough spatial structure.

4. **Neuron probing (all 128 neurons):** For each bottleneck neuron, optimized an input image from random noise (300 iterations of gradient ascent) to maximally activate that single neuron. This reveals what each neuron has learned to detect.

5. **Automatic classification of neurons:** Classified each neuron as "COLOUR" or "SHAPE" based on spatial standard deviation of its preferred image. Low spatial variation (< 0.15) means the neuron responds to flat/solid colors. High variation means it responds to textures, edges, or shape patterns.

6. **Side-by-side comparison:** Displayed the top colour neurons vs. top shape neurons in a clean grid, sorted by mean activation.

7. **Mean activation bar chart:** Plotted the average activation of each hidden unit across the dataset.

**Key findings:**

- A significant fraction of the 128 bottleneck neurons are classified as COLOUR neurons -- they respond to flat, uniform color patches corresponding to the background colors in the dataset.
- The remaining neurons show textured or edge-like patterns (SHAPE neurons), but these are less dominant.
- This confirms that the lazy model's hidden states explicitly encode background color as distinct, separable features. The spurious correlation is not just a diffuse bias -- it is crystallized into dedicated internal representations.
- The SAE successfully decomposes the hidden states into interpretable colour vs. shape directions, demonstrating that mechanistic interpretability techniques can identify and isolate shortcut features in neural networks.

---

## Compute and Runtime Notes

- All training was done on Kaggle notebooks with GPU T4 runtime.
- 25% stratified subsets were used to keep training times manageable (approximately 12,000 training images per run).
- DataLoaders used `num_workers=2` and `pin_memory=True` throughout for efficient GPU data transfer.
- Reproducibility seeds are set (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`) in every notebook.

---

## References

- Selvaraju, R.R. et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017. [arXiv:1610.02391](https://arxiv.org/pdf/1610.02391)
- Andrej Karpathy, ConvNetJS -- browser-based neural network visualizations.
- NPTEL Deep Learning Lectures.
- PyTorch Documentation (transforms, DataLoader, ResNet).
- Kaggle platform for compute and dataset hosting.

---

## What Was Completed

| Task | Status | Description |
|------|--------|-------------|
| Task 0 | Done | Colored-MNIST dataset with textured backgrounds, 95/5 bias, published on Kaggle |
| Task 1 | Done | LazyCNN + ResNet-18 trained and evaluated, confusion matrices, color-bias proof |
| Task 2 | Done | Filter visualization, feature maps, embedding inversion, deep dreaming, neuron probing (3 modes), polysemanticity exploration on both models |
| Task 3 | Done | Grad-CAM from scratch, validated against library, applied to both models on biased and conflicting images |
| Task 4 | Done | Two intervention methods (BatchNorm, BatchNorm + color gradient penalty), hard test accuracy raised from 7% to 84% |
| Task 5 | Done | Three targeted attacks (FGSM, PGD, DeepFool), compared lazy vs robust model, documented failed universal attack attempt |
| Task 6 | Done | Sparse Autoencoder on pixel space, probed all 128 neurons, auto-classified colour vs shape features, confirmed colour encoding in hidden states |
