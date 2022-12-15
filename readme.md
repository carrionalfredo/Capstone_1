# Machine Learning Zoomcamp 2022
## Capstone 1 - Pistachio Image Classifier

The present project was elaborated as the Capstone 1 project, for the [Machine Learning Zooncamp 2022](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) realized by [Datacamp.club](https://datatalks.club/).

The objective of this project, is to develop an machine learning model for classificate images of pistachios, and determine to wich variety belongs.

For this, I propose to build and train a deep learning model, taking a dataset of pistachios varieties as training data for the model.

## Pistachios
![alt text](https://github.com/carrionalfredo/Capstone_1/blob/main/images/readme/640px-Pistachio_vera.jpg)
[From THOR - Pistachio, CC BY 2.0](https://commons.wikimedia.org/w/index.php?curid=40606682)

The pistachio (Pistacia vera), a member of the cashew family, is a small tree originating from Central Asia and the Middle East. The tree produces seeds that are widely consumed as food.

### Pistachio varieties

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

The dataset used in this project is a dataset of 2148 images of pistachios, 1232 of Kirmizi type and 916 of Siirt type. This dataset can be found in [Visualdata.io](https://visualdata.io/discovery/dataset/906f860910230c325f1fa63da88f6c847a06724a)

![alt text](https://www.mdpi.com/electronics/electronics-11-00981/article_deploy/html/images/electronics-11-00981-g001.png)
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
