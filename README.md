# Knee Arthritis Detection using Deep Learning

This project focuses on the automatic detection and grading of **knee osteoarthritis** using X-ray images and Convolutional Neural Networks (CNN). The goal is to classify knee conditions into **five severity levels** (KL grades 0 to 4) based on radiographic features.

---

## 📌 What This Project Does

- Takes knee X-ray images as input  
- Processes and prepares them for learning  
- Trains a CNN model to classify the **KL grade** (severity of arthritis)  
- Improves model accuracy through **preprocessing**, **data augmentation**, and **model tuning**  
- Evaluates model performance on test images  

---

## 📂 Dataset Overview

- The dataset contains X-ray images of knees, labeled with **KL grades** from 0 to 4:
  - 0: Normal  
  - 1–2: Mild to moderate arthritis  
  - 3–4: Severe arthritis

- Each image is:
  - Resized to **256×256**
  - Converted to **grayscale**
  - Normalized for consistent pixel values

---

## 🔧 Model Development Process

### 🏁 Initial Model
- A basic CNN was built and trained directly on the raw X-ray images.
- Architecture included `Conv2D`, `MaxPooling`, `Dropout`, and `Dense` layers.
- Result: **~35% accuracy**  
  - Reason: model was distracted by irrelevant image regions (not focused on knee joint).

---

### 🚀 Improved Approach

To improve the performance, the following strategies were applied:

#### 1. **Focused Input**
- Cropped the **region of interest (ROI)** — only the knee joint area was used.
- This helped the model concentrate on the actual affected zone.

#### 2. **Contrast Enhancement**
- Applied **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to make bone structures and joint gaps more visible.

#### 3. **Data Augmentation**
- Introduced random:
  - Flipping
  - Rotation
  - Zoom
  - Brightness variation  
- This created more diverse training examples and reduced overfitting.

#### 4. **Deeper CNN Architecture**
- Added more convolutional layers and increased the number of filters.
- Introduced dense layers: `Dense(1024)` → `Dense(512)` before final output.
- Applied dropout for regularization and better generalization.

#### 5. **Longer & Smarter Training**
- Trained over **300 epochs**
- Used `ModelCheckpoint` to save the best performing model during training

---

## 📈 Results Summary

| Stage                   | Description                                 | Accuracy   |
|------------------------|---------------------------------------------|------------|
| Initial Model          | Basic CNN on raw images                     | ~35%       |
| With Augmentation      | Added data augmentation + ROI + CLAHE       | ~68%       |
| Final Improved Model   | Deep CNN + Dense Layers + Full Preprocessing| **~78%**   |

---

## ✅ What Helped Improve Accuracy

- **Focused images**: Cropping the knee joint removed distractions  
- **Contrast boost**: CLAHE highlighted joint spaces more clearly  
- **Augmentation**: Helped the model learn better from fewer examples  
- **Model depth**: Deeper layers extracted better features  
- **Regularization**: Dropout reduced overfitting  
- **Long training**: More epochs with checkpoints improved learning stability

---

## 🧪 Final Notes

- The model now accurately classifies the severity of knee arthritis into 5 categories.
- It is trained purely on grayscale X-ray images using supervised learning.
- With medical explainability tools like heatmaps, the model’s decision process can be visualized in future versions.

---

# Knee Arthritis Detection using Convolutional Neural Networks
### Technical Reference Document — System Architecture, Execution Protocol, and Terminological Compendium

---

## Table of Contents

1. System Overview
2. Computational Environment and Dependencies
3. Dataset Specification
4. Execution Protocol
5. Foundational Terminology
6. Intermediate Concepts
7. Advanced Architectural Concepts
8. Model Architecture Progression
9. Persistent Artifacts
10. Performance Interpretation

---

## 1. System Overview

This system implements a deep learning pipeline for automated classification of knee osteoarthritis severity from radiographic (X-ray) images. Classification targets five discrete severity grades conforming to the Kellgren-Lawrence grading scale, a clinically established standard for osteoarthritis staging:

- Grade 0: No radiographic features of osteoarthritis
- Grade 1: Doubtful joint space narrowing with possible osteophytic lipping
- Grade 2: Definite osteophytes and possible narrowing of joint space
- Grade 3: Moderate multiple osteophytes, definite joint space narrowing, some sclerosis, and possible deformity of bone ends
- Grade 4: Large osteophytes, marked narrowing of joint space, severe sclerosis, and definite deformity of bone ends

The implementation follows a controlled, incremental experimental methodology. Three distinct model architectures are constructed and evaluated sequentially. Each successive model introduces one architectural or training modification over the preceding iteration, thereby permitting rigorous attribution of performance gains to specific design decisions. The channel attention mechanism common to all three models is the Squeeze-and-Excitation (SE) block, which enables adaptive feature recalibration across the channel dimension of learned representations.

---

## 2. Computational Environment and Dependencies

The system is designed for execution within the Google Colaboratory (Colab) cloud-based interactive Python environment. Colab provides browser-accessible compute resources including GPU acceleration without requiring local software installation.

Prior to execution, GPU acceleration must be explicitly enabled via the runtime configuration interface: Runtime > Change runtime type > Hardware accelerator > GPU (T4). Training without GPU acceleration is computationally impractical for the epoch counts employed.

Dataset acquisition requires a Kaggle account and a valid Kaggle API credential file (kaggle.json). Instructions for credential acquisition are provided in Section 4.

The following libraries are required and are pre-installed within the Colab environment:

- TensorFlow / Keras: construction, training, serialization, and inference of neural network models
- NumPy: numerical array computation and tensor manipulation
- Matplotlib: visualization of training dynamics and radiographic sample images

---

## 3. Dataset Specification

The dataset used is the "Annotated Dataset for Knee Arthritis Detection" (Kaggle identifier: hafiznouman786/annotated-dataset-for-knee-arthritis-detection). Upon extraction, the dataset produces a directory named Training containing five subdirectories designated 0 through 4, each corresponding to one Kellgren-Lawrence grade and containing the radiographic images of that grade.

All images are loaded in grayscale mode at a uniform resolution of 256 by 256 pixels, yielding input tensors of shape (256, 256, 1).

The dataset is partitioned into three non-overlapping subsets using a sequential splitting strategy:

- A primary split reserves 10% of the full dataset as the held-out test partition.
- A secondary split reserves 10% of the remaining 90% as the validation partition.
- The residual data constitutes the training partition.

This yields approximate proportions of 81% training, 9% validation, and 10% test. The test partition is strictly withheld from all model parameter updates and validation monitoring; it is used exclusively for final performance reporting.

---

## 4. Execution Protocol

All components of the system must be executed sequentially in the order prescribed below. Each stage depends on the successful completion of all preceding stages.

### Stage 1 — GPU Runtime Activation

Navigate to Runtime > Change runtime type. Set Hardware accelerator to GPU. Confirm and save. This must be completed before any execution commences.

### Stage 2 — Kaggle API Credential Configuration

Obtain the Kaggle API key by logging into kaggle.com, navigating to Account settings, and selecting "Create New API Token" under the API section. This action generates and downloads the file kaggle.json to the local machine.

Upload kaggle.json to the Colab file system via the Files panel (left sidebar). Once uploaded, the credential setup sequence executes the following commands to install the key in the location expected by the Kaggle CLI:

```
!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### Stage 3 — Dataset Acquisition and Extraction

The Kaggle CLI command downloads the dataset archive from the Kaggle platform into the active working directory:

```
!kaggle datasets download -d hafiznouman786/annotated-dataset-for-knee-arthritis-detection
```

The downloaded archive is then silently extracted:

```
!unzip -qq annotated-dataset-for-knee-arthritis-detection.zip
```

Following extraction, the directory Training/ with five subdirectories (0, 1, 2, 3, 4) will be present at /content/Training/.

### Stage 4 — Library Importation

All required Python modules are imported at this stage. These include the Keras dataset loading utility (image_dataset_from_directory), the dataset splitting utility (split_dataset), the neural network layer library (layers), the Keras backend (K), the top-level keras API, Matplotlib's pyplot interface, and NumPy. All subsequent stages depend on the availability of these imports.

### Stage 5 — Attention Mechanism Definition

The Squeeze-and-Excitation block is defined as the function se_block(input_tensor, ratio=16). This function must be registered in the runtime environment before any model construction commences, as all three model architectures invoke it during graph construction. The mechanism is described in full in Section 7.

### Stage 6 — Dataset Loading and Visual Verification

The Training directory is ingested using image_dataset_from_directory with grayscale color mode, 256x256 resolution, and unbatched loading (batch_size=None). The first twenty samples are rendered in a 4-by-5 grid using the inferno colormap to provide a visual verification that image loading and label assignment are functioning correctly.

### Stage 7 — Dataset Partitioning

The full dataset is split sequentially into training, validation, and test partitions as described in Section 3. The cardinality of each partition is printed to confirm expected proportions.

### Stage 8 — Model 1: Baseline Architecture Construction and Training

The first model is constructed using paired convolutional blocks at four depth levels (filter counts 32, 64, 128, 256 — each applied in two consecutive convolutional layers before pooling), followed by the SE attention block, a flattening operation, Dropout with rate 0.5, and a five-class softmax output layer. Input pixel values are rescaled from [0, 255] to [0.0, 1.0] within the model graph.

The model is compiled with the RMSProp optimizer and sparse categorical crossentropy loss. Training proceeds for 20 epochs with batch size 32 and no model checkpointing. This establishes a reference performance benchmark.

Upon completion of training, the show_plots utility function (which accepts the history dictionary returned by model.fit and renders accuracy and loss curves for both training and validation partitions) is defined and invoked to display training dynamics.

### Stage 9 — Model 2: Augmented Architecture Construction and Training

A second model is constructed with a lighter convolutional backbone: five single-convolution stages with progressively growing filter counts (8, 16, 32, 64, 128), each followed by MaxPooling, succeeded by the SE attention block, Flatten, Dropout(0.5), and the softmax output layer. A data augmentation pipeline (random horizontal flip, random rotation up to 36 degrees, random zoom up to 20%) is prepended to the model graph and applied exclusively during training.

The model is compiled with the same optimizer and loss as Model 1. Training proceeds for 100 epochs with batch size 32. A ModelCheckpoint callback monitors validation accuracy and persists the best-performing weights to the file best_cnn_with_data_augmentation.h5.

Following training, the peak validation accuracy is reported, the best-checkpoint model is reloaded, and final performance is evaluated on the test partition.

### Stage 10 — Model 3: Dense-Head Architecture Construction and Training

The third and final model extends the augmented architecture of Model 2 with an additional MaxPooling layer after the fifth convolutional stage and a deeper classification head: Dropout(0.2), Dense(1024, ReLU), Dropout(0.2), Dense(512, ReLU), Dropout(0.2), and the softmax output layer. The data augmentation pipeline is preserved.

Training proceeds for 300 epochs with batch size 16. A ModelCheckpoint callback persists the best-performing weights to the file best_cnn_with_data_augmentation_and_dense.h5.

Following training, the peak validation accuracy is reported, the model architecture summary is printed, training curves are visualized, and a formatted final accuracy summary is printed displaying both final training accuracy and best validation accuracy as percentages.

### Stage 11 — Qualitative Inference Verification

Five samples are drawn at random from the test partition. For each sample, the model produces a grade prediction, which is compared to the ground-truth label. The radiographic image is rendered alongside the actual and predicted grade designations. A micro-accuracy figure (correct predictions out of five) is computed and printed. This stage functions as a qualitative sanity check and is not intended as a statistically valid performance measure.

---

## 5. Foundational Terminology

### Radiographic Image as a Numerical Tensor

In the context of this system, a radiographic image is not represented as a visual file during processing. Upon loading, it is converted to a tensor — a structured multi-dimensional array of numerical values. A grayscale image of 256x256 pixels yields a tensor of shape (256, 256, 1), representing 256 spatial rows, 256 spatial columns, and a single intensity channel. Pixel values are integers in the range [0, 255], where 0 represents black (minimum radiodensity) and 255 represents white (maximum radiodensity).

### Class Label

Each radiographic sample is associated with an integer label in {0, 1, 2, 3, 4} denoting its Kellgren-Lawrence grade. The supervised learning objective is to train a model to map input tensors to correct class labels.

### Mini-Batch

Neural network parameters are updated not after each individual sample but after processing a group of samples simultaneously, termed a mini-batch. Models 1 and 2 employ a batch size of 32; Model 3 employs a batch size of 16. Smaller batch sizes provide noisier gradient estimates but update parameters more frequently per epoch, which can benefit convergence during prolonged training.

### Training Epoch

One epoch corresponds to a complete pass through the entirety of the training partition. Model 1 trains for 20 epochs; Model 2 for 100 epochs; Model 3 for 300 epochs.

### Loss Function

The loss function quantifies the discrepancy between predicted class probability distributions and ground-truth labels. All three models employ sparse categorical crossentropy, which computes the logarithmic divergence between the predicted probability of the correct class and 1.0. This formulation accepts integer labels directly without requiring one-hot encoding.

### Classification Accuracy

The proportion of input samples for which the class with the highest predicted probability matches the ground-truth label. Reported after each training epoch as accuracy (training partition) and val_accuracy (validation partition).

### Optimizer

The optimization algorithm responsible for computing parameter updates from computed gradients. All three models employ the RMSProp (Root Mean Square Propagation) optimizer, which maintains a moving average of squared gradients and scales updates accordingly, providing stable convergence on non-stationary objectives.

---

## 6. Intermediate Concepts

### Input Rescaling

All three models include a Rescaling layer as the first processing step (applied after data augmentation where present). This layer divides all pixel intensity values by 255.0, mapping the input range [0, 255] to the normalized range [0.0, 1.0]. Neural network training exhibits significantly improved stability and convergence when input values are small and bounded.

### Convolutional Layer

A two-dimensional convolutional layer (Conv2D) is the primary feature extraction operator. A small learnable filter matrix of specified kernel size slides across the spatial dimensions of an input feature map, computing a weighted inner product at each position. Each filter learns to detect a specific local pattern — such as an intensity edge, a textural gradient, or a structural motif — within the radiographic image. Multiple filters operate in parallel, each producing a distinct output channel in the resulting feature map. As filters deepen through successive layers, they detect progressively more abstract and semantically meaningful structures.

Model 1 employs paired convolutions (two consecutive convolutional layers with identical filter counts) at each depth level, a design pattern associated with the VGG family of architectures, which builds deeper representations prior to spatial downsampling. Models 2 and 3 employ a single convolution per depth level, substantially reducing parameter count while preserving representational capacity when combined with data augmentation.

### Spatial Downsampling via MaxPooling

Following convolutional feature extraction, MaxPooling reduces the spatial resolution of the feature map by a factor of two in each spatial dimension by retaining only the maximum activation value within each non-overlapping 2x2 region. This operation reduces computational cost, introduces a degree of translational invariance, and prevents overfitting by constraining the spatial precision to which features must be localized.

### ReLU Activation Function

The Rectified Linear Unit (ReLU) activation function, applied elementwise after each convolutional layer, maps negative activations to zero while preserving positive activations unchanged: f(x) = max(0, x). This piecewise linear nonlinearity introduces the representational non-linearity required for deep networks to approximate complex, non-convex functions of the input.

### Softmax Classification Head

The terminal Dense layer in all three models applies the softmax activation function to produce a valid probability distribution over the five Kellgren-Lawrence classes. Softmax exponentiates each raw logit and normalizes by their sum, ensuring all outputs are strictly positive and sum to unity. The class index with the maximum softmax output constitutes the model prediction.

### Dropout Regularization

Dropout is a stochastic regularization technique that, during training, independently zeroes each neuron's activation with a specified probability at each forward pass. Model 1 and Model 2 apply Dropout with rate 0.5 immediately preceding the output layer. Model 3 applies Dropout with rate 0.2 between each Dense layer in the classification head. By preventing co-adaptation of neurons, dropout imposes an ensemble-like inductive bias and reduces overfitting, particularly in the high-capacity fully connected portion of the network.

### Overfitting and Generalization

Overfitting occurs when a model learns to reproduce idiosyncratic patterns of the training data that do not generalize to unseen samples. In training curves, this manifests as continued improvement of training accuracy paired with stagnation or degradation of validation accuracy. This system addresses overfitting progressively: Model 2 introduces data augmentation to increase effective training data diversity; Model 3 additionally reduces the dropout rate and deepens the classification head, trading individual layer capacity for architectural depth.

### Data Augmentation

Data augmentation synthetically expands the effective training distribution by applying stochastic geometric and photometric transformations to training images prior to each forward pass. The augmentation pipeline employed in Models 2 and 3 comprises three operations:

- RandomFlip("horizontal"): horizontally mirrors the image with probability 0.5, exploiting the bilateral symmetry of knee radiographs
- RandomRotation(0.1): applies a random rotation uniformly sampled from ±36 degrees, accommodating variation in patient positioning
- RandomZoom(0.2): applies a random isotropic zoom factor uniformly sampled from the range [0.8, 1.2], accommodating variation in imaging distance

Augmentation transformations are applied stochastically during training only; validation and test partitions are evaluated on unaugmented images.

### Fully Connected (Dense) Layer

A Dense layer establishes a complete bipartite connection between all input units and all output units, with each connection parameterized by a learned weight. In the classification heads of Models 1 and 2, a single Dense layer with five outputs and softmax activation produces the final class distribution. Model 3 interposes two intermediate Dense layers (1024 and 512 units, both with ReLU activation) between the flattened feature representation and the output layer, allowing the model to learn higher-order, non-linear combinations of the spatial features prior to classification.

### ModelCheckpoint Callback

The ModelCheckpoint callback, registered during model training, monitors the val_accuracy metric after each epoch. Whenever a new maximum validation accuracy is observed, the complete model weights are serialized to disk. This mechanism ensures that the best-performing model configuration encountered during the full training run is preserved regardless of subsequent degradation due to overfitting. At evaluation time, the serialized checkpoint is reloaded, yielding performance figures associated with the optimal training epoch rather than the terminal epoch.

### Training Dynamics Visualization (show_plots)

The show_plots function accepts the history dictionary returned by model.fit, which records per-epoch scalar metrics across the training run, and renders two graphical representations: an accuracy curve plotting training and validation accuracy against epoch index, and a loss curve plotting training and validation loss against epoch index. Divergence between the training and validation curves is the primary diagnostic signal for overfitting.

---

## 7. Advanced Architectural Concepts

### Squeeze-and-Excitation (SE) Block

Implemented as se_block(input_tensor, ratio=16).

In a standard convolutional network, all feature channels within a given layer are propagated forward with equal weighting, regardless of their relevance to the classification target. The Squeeze-and-Excitation block introduces a learned, data-driven channel recalibration mechanism that selectively amplifies diagnostically informative channels and suppresses those of lower relevance. It operates in two sequential phases.

The Squeeze phase performs global spatial aggregation. GlobalAveragePooling2D computes the spatial mean of each channel independently, collapsing a feature map of shape (H, W, C) into a descriptor vector of length C. Each element of this vector encodes the global average activation of one feature channel across the entire spatial extent of the feature map — a compact summary of how strongly each learned feature type is expressed in the current input. The descriptor is subsequently reshaped to (1, 1, C) to maintain dimensional compatibility with subsequent Dense operations.

The Excitation phase produces a channel-wise attention weighting. Two fully connected layers are applied sequentially to the squeezed descriptor. The first Dense layer reduces the channel dimension from C to C/ratio (with ratio=16, this is C/16) using ReLU activation, forming a dimensionality-reduced intermediate representation that encodes learned inter-channel dependencies. The second Dense layer restores the dimension to C using sigmoid activation, mapping the intermediate representation to a vector of per-channel attention weights in the range (0, 1).

The attention weights are applied to the original feature map via elementwise multiplication (Multiply). Each channel of the feature map is independently scaled by its corresponding attention weight. Channels with weights approaching 1.0 are transmitted nearly unmodified; channels with weights approaching 0.0 are substantially suppressed. The output is thus a recalibrated feature map in which the network has learned to direct representational emphasis toward the most diagnostically relevant feature channels for osteoarthritis classification.

The compression ratio parameter (ratio=16) governs the dimensionality of the excitation bottleneck. The value 16 follows the prescription of the original Squeeze-and-Excitation Networks publication (Hu et al., 2018) and represents an empirically validated trade-off between the expressivity of inter-channel modelling and the computational overhead of the attention branch.

### Paired Convolutional Architecture (Model 1)

Model 1 constructs its convolutional backbone as four sequential paired blocks, each comprising two Conv2D layers with identical filter counts followed by MaxPooling. The filter progression is: Conv(32), Conv(32), Pool, Conv(64), Conv(64), Pool, Conv(128), Conv(128), Pool, Conv(256), Conv(256). This design, adapted from the VGG architectural family, constructs deep representations at each spatial resolution scale before committing to spatial downsampling. The result is a higher-capacity feature representation at the cost of a substantially larger parameter count relative to single-convolution architectures.

### Progressive Single-Convolution Architecture (Models 2 and 3)

Models 2 and 3 replace paired convolution blocks with single convolutions, one per depth level, each immediately followed by MaxPooling. Filter counts increase progressively: Conv(8), Pool, Conv(16), Pool, Conv(32), Pool, Conv(64), Pool, Conv(128). This architecture has substantially fewer parameters than Model 1 and, when trained with data augmentation, exhibits superior generalization on limited medical imaging datasets where the risk of overfitting is elevated.

### Additional Downsampling Stage in Model 3

Model 3 appends a fifth MaxPooling operation after Conv(128), a stage absent in Model 2. This further reduces the spatial dimensions of the feature map prior to flattening, decreasing the input dimensionality to the classification head and rendering the subsequent Dense layers computationally tractable without parameter explosion.

### Deep Classification Head (Model 3)

After the convolutional backbone and SE block, the feature map is vectorized by a Flatten operation. In Models 1 and 2, this vector is passed directly through a single Dropout layer to the softmax output. In Model 3, the vector traverses a three-stage classification head: Dropout(0.2), Dense(1024, ReLU), Dropout(0.2), Dense(512, ReLU), Dropout(0.2), followed by the softmax output. The two intermediate Dense layers learn non-linear composite representations of the spatial features extracted by the convolutional backbone, enabling the model to form more complex decision boundaries in the classification space.

### Sparse Categorical Crossentropy

Sparse categorical crossentropy is the loss function employed throughout all three models. For a predicted probability vector p and a scalar integer ground-truth label y, the loss is computed as -log(p_y), the negative log-likelihood of the correct class under the predicted distribution. It is mathematically equivalent to categorical crossentropy applied to one-hot encoded targets but is computationally preferable when labels are provided as integers rather than one-hot vectors, as is the case in this implementation where labels are integer grades 0 through 4.

---

## 8. Model Architecture Progression

### Model 1 — Baseline Reference Architecture

Convolutional backbone: Four paired convolutional blocks (filter counts 32, 64, 128, 256; two consecutive convolutions per block) followed by MaxPooling after each block.
Attention: SE block applied after the final convolutional block.
Classification head: Flatten, Dropout(0.5), Dense(5, softmax).
Training configuration: 20 epochs, batch size 32, RMSProp, no data augmentation, no checkpoint persistence.
Purpose: Establishes a quantitative baseline for comparison with subsequent models.

### Model 2 — Augmented Lightweight Architecture

Convolutional backbone: Five single convolutional stages (filter counts 8, 16, 32, 64, 128), each followed by MaxPooling, with the exception that no MaxPooling follows the fifth stage.
Attention: SE block applied after the fifth convolutional stage.
Classification head: Flatten, Dropout(0.5), Dense(5, softmax).
Data augmentation: RandomFlip, RandomRotation(0.1), RandomZoom(0.2) applied at input during training.
Training configuration: 100 epochs, batch size 32, RMSProp, ModelCheckpoint monitoring val_accuracy.
Serialized artifact: best_cnn_with_data_augmentation.h5

### Model 3 — Augmented Architecture with Dense Classification Head (Primary System)

Convolutional backbone: Five single convolutional stages (filter counts 8, 16, 32, 64, 128), each followed by MaxPooling including after the fifth stage.
Attention: SE block applied after the fifth MaxPooling layer.
Classification head: Flatten, Dropout(0.2), Dense(1024, ReLU), Dropout(0.2), Dense(512, ReLU), Dropout(0.2), Dense(5, softmax).
Data augmentation: RandomFlip, RandomRotation(0.1), RandomZoom(0.2) applied at input during training.
Training configuration: 300 epochs, batch size 16, RMSProp, ModelCheckpoint monitoring val_accuracy.
Serialized artifact: best_cnn_with_data_augmentation_and_dense.h5

---

## 9. Persistent Artifacts

Two model serialization files are produced during training and written to the active working directory:

best_cnn_with_data_augmentation.h5 — contains the Model 2 weights corresponding to the epoch of maximum validation accuracy observed across the 100-epoch training run.

best_cnn_with_data_augmentation_and_dense.h5 — contains the Model 3 weights corresponding to the epoch of maximum validation accuracy observed across the 300-epoch training run. This is the primary trained system and is used for final test evaluation and qualitative inference verification.

To retrieve these files from the Colab environment, locate them in the Files panel, right-click, and select Download.

---

## 10. Performance Interpretation

### Training and Validation Curves

The show_plots utility renders two diagnostic plots. The accuracy plot displays per-epoch training accuracy and validation accuracy over the full training run. Healthy generalization is indicated by both curves rising and converging toward a common asymptote. Overfitting is indicated by training accuracy continuing to rise while validation accuracy plateaus or declines. The loss plot presents the same diagnostic in terms of the loss metric, where divergence of validation loss from training loss is the canonical signal of overfitting.

### Test Partition Evaluation

model.evaluate applied to the held-out test partition yields test loss and test accuracy. Test accuracy is the most externally valid performance measure in this system, as the test partition has been strictly withheld from all training and validation monitoring operations. The best-checkpoint model (loaded from the .h5 serialization) is used for this evaluation.

### Peak Validation Accuracy

max(history.history["val_accuracy"]) returns the maximum validation accuracy recorded at any epoch during the training run. Because the ModelCheckpoint callback persists model weights at the epoch corresponding to this maximum, reloading the serialized model and evaluating on the test partition yields performance attributable to this optimal configuration rather than the terminal training state.

### Qualitative Sample Inference

The inference verification procedure randomly selects five samples from the test partition, applies the loaded model to each, and displays the radiographic image alongside its actual and predicted Kellgren-Lawrence grade. A micro-accuracy (correctly classified samples out of five) is computed and reported. Due to the small sample size, this figure carries high statistical variance and is intended solely as a qualitative illustration of system behavior rather than as a quantitative performance estimate.

