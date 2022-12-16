# Machine Learning Zoomcamp 2022
## Capstone 1 - Pistachio Image Classifier

The present project was elaborated as the Capstone 1 project, for the [Machine Learning Zooncamp 2022](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) realized by [Datacamp.club](https://datatalks.club/).

The objective of this project, is to develop an machine learning model for classificate images of pistachios, and determine to wich variety belongs.

For this, I propose to build and train a deep learning model, taking a dataset of pistachios varieties as training data for the model.

## Pistachios
![alt text](https://github.com/carrionalfredo/Capstone_1/blob/main/images/640px-Pistachio_vera.jpg)
[From THOR - Pistachio, CC BY 2.0](https://commons.wikimedia.org/w/index.php?curid=40606682)

The pistachio (Pistacia vera), a member of the cashew family, is a small tree originating from Central Asia and the Middle East. The tree produces seeds that are widely consumed as food.

### Pistachio Varieties

They are classified according to their origin; their colour and size differentiates them. The main types of pistachio are the pistachio of Sicily, which is green, the smaller Tunis and the Pistachio of Levante, yellow and therefore less appreciated commercially.

The main types found in the market are the Noble or Sicily pistachio, with a green and very appreciated almond; Tunis, smaller but equally appreciated, and the Levante, with a yellow edible part and less accepted because of its flavour.

Usually, the pistachio varieties are classified according to their place of origin or culture. Each country has its own selections, whose main differences are the colour, flavour, size, period of harvesting and qualities.

Some varieties of pistachios

" Kerman"
Pistachio nut of great size and good quality. Selected in Iran, it was introduced in the U.S.A. and it is also cultivated in Spain (in Castilla-La Mancha) where the fruit ripens during the first fortnight of September.

" Peter"
It is used as a male cultivar with Kerman it has a good polen production and they partly coincide during the flowering period. Selected in California.

" Uzun"
Pistachio nut of average size, long and clear green. It is cultivated in Turkey.

" Kirmizi ‘
Pistachio nut of average size and reddish colour. Along with the cultivar Uzum, it is the most cultivated variety in Turkey.

" Abiad miwahi ‘
Pistachio nut of average size, white colour and excellent quality. Cultivated in Turkey.

" Achouri ‘
Pistachio nut of average size, red colour, excellent quality and very productive. Cultivated in Syria.

" Batouri ‘
Thick fruit of whitish colour and good quality. Important cultivar in Syria.

" Sefideh-Montaz" and " Imperiale de Dameghan"
The fruit of these varieties is round, thick and yellowish. Very appreciated in Iran.

" Kouchka ‘
Quite thick pistachio, cream white colour and good quality.

" Mateur"
Long fruit, average size, yellow greenish colour and good taste quality. It was selected in Tunis and it gives good results in Spain. In Castilla-La Mancha it ripens at the end of August.

" Larnaka ‘
Average size pistachio, less long than ‘Mateur ". Original from Cyprus. It is cultivated in Greece and in Spain, giving good results.

" Aegina ‘
Medium size fruit, long and similar to " Mateur ". It comes from Greece and it also gives good results in Spain.

[Source](https://www.frutas-hortalizas.com/Fruits/Types-varieties-Pistachio-nut.html)

## Dataset

The dataset used in this project is a dataset of 2148 600x600px jpeg images of pistachios, 1232 of Kirmizi type and 916 of Siirt type. This dataset can be found in [Visualdata.io](https://visualdata.io/discovery/dataset/906f860910230c325f1fa63da88f6c847a06724a)

![Dataset source](https://www.mdpi.com/electronics/electronics-11-00981/article_deploy/html/images/electronics-11-00981-g001.png)
[source: Visualdata.io](https://visualdata.io/discovery/dataset/906f860910230c325f1fa63da88f6c847a06724a)

The data set is organized of the following way:
```
Pistachio_Image_Dataset/Pistachio_Image_Dataset
├── Kirmizi_Pistachio
└── Siirt_Pistachio
```

<figure>
  <img
  src="https://github.com/carrionalfredo/Capstone_1/blob/main/images/kimizi_images_dataset.jpg"
  alt="Kimizi images in dataset folder."
  title="Kimizi images dataset">
  <figcaption>Kimizi images dataset</figcaption>
</figure>


<figure>
  <img
  src="https://github.com/carrionalfredo/Capstone_1/blob/main/images/siirt_images_dataset.jpg"
  alt="Siirt images in dataset folder."
  title="Siirt images dataset">
  <figcaption>Siirt images dataset</figcaption>
</figure>

## Preparation of the dataset

The preparation of the dataset consists of the following steps:

```python
ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10.0,
    zoom_range=0.1,
    vertical_flip=True
    )
```

```python
image_gen.flow_from_directory(
    './Pistachio_Image_Dataset/Pistachio_Image_Dataset/',
    target_size=(150,150),
    batch_size=32
    )
```

After that, the datat set consist of:
```
Found 2148 images belonging to 2 classes.
{'Kirmizi_Pistachio': 0, 'Siirt_Pistachio': 1}
````

Then split the data set in train, validation and test sets, with a distribution of 60%, 20%, and  20%.

## Build of the base model

The base model consist of:
- Conv2D input layer.
- MaxPool2D hidden layer.
- Dropout hidden layer.
- Flatten hidden layer.
- Dense hidden layer.
- Dense output layer.

Defined by:
```python
model.add(Conv2D(32,3,3, input_shape = (150,150,3), activation = 'relu'))

  model.add(MaxPool2D(2,2))

  model.add(Dropout(droprate, seed=1))

  model.add(Flatten())

  model.add(Dense(128, activation='relu'))

  model.add(Dense(2, activation='softmax', name = 'output'))
```

For ompilation of the model, is used ``Adam`` optimizer and the loss function is ``CategoricalCrossentropy``.

!['Base model](https://github.com/carrionalfredo/Capstone_1/blob/main/images/base_model.png)

## Training of the model

### Training of the base model

The base model was trained with the following hyperparameters:
```python
neurons=32
droprate = 0.5
learning_rate=0.001
batch_size = 32
epochs = 100
```
The training & validation accuracy and loss values obtained are the following:

![Base model](https://github.com/carrionalfredo/Capstone_1/blob/main/images/Base_model_results.png)

The evaluation of the base model with the test data are:

```
Test loss:  1.506 Test accuracy:  0.4286
```
### Hyperparameters tuning

In order to improve the accuracy and reduce the loss values, the parameters ```learning_rate``` and ```droprate``` were tuned. After that process, the final model parameters are:
- ```learning_rate = 0.0001```.
- ```droprate = 0.6```.

The training & validation accuracy and loss values obtained for the final model are the following:

![Final model](https://github.com/carrionalfredo/Capstone_1/blob/main/images/Final_model_results.png)

Finally, the summary of this final model is next.

````
Model: "Final_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_20 (Conv2D)          (None, 50, 50, 32)        896       
                                                                 
 max_pooling2d_20 (MaxPoolin  (None, 25, 25, 32)       0         
 g2D)                                                            
                                                                 
 dropout_20 (Dropout)        (None, 25, 25, 32)        0         
                                                                 
 flatten_20 (Flatten)        (None, 20000)             0         
                                                                 
 dense_20 (Dense)            (None, 128)               2560128   
                                                                 
 output (Dense)              (None, 2)                 258       
                                                                 
=================================================================
Total params: 2,561,282
Trainable params: 2,561,282
Non-trainable params: 0
_________________________________________________________________
````
For this configration of the final model, the evaluation with test data are the following:

````
Test loss:  0.8363 Test accuracy:  0.4286
````
