																		# Data Science Portfolio - Riaz Ahmed. T
Repository containing portfolio of data science projects completed by me for academic and self learning purposes. They are Presented in the form of Jupyter notebooks, PDF files and R markdown files (published at RPubs).

## Instructions for Running Python Notebooks Locally
1. Install dependencies using requirements.txt.
2. Run notebooks as usual by using a jupyter notebook server, Vscode etc.

## Contents

- ### Machine Learning

	- [House price prediction using XGBoost and ANN models](https://github.com/riazahmedtgithub/Models/blob/main/HousePrediction%20using%20XGBoost%20and%20ANN.pdf): A model to predict the value of a given house in the real estate market.
		- Data has been sourced from Kaggle (n.d.) Housing Prices Competition for Kaggle Learn Users Retrieved September 6, 2024, from https://www.kaggle.com/c/home-data-for-ml-course/data
		- Feature creation, selection and elimination.
		- Visualization using frequency distribution, box plots, bar charts, correlation maxtrix etc.
		- Data preparation.
		- Building XGBoost and ANN models.
		- Parameter tuning using GridSearchCV.
		- Feature elimination using RFECV. 
		- Hyperparameter tuning with RandomizedSearchCV for Kerasregressor.
  		- Model metrics comparison and final model recommendation.
    	
	- [Predicting Neurodegenerative disease using KNN and PCA](https://github.com/riazahmedtgithub/Models/blob/main/Predicting%20NeuroDegenerative%20disease%20using%20KNN%20and%20PCA.ipynb): To predict the neurodegenerative disease based on the test results of Albumin, Sodium, Urine PH, Haemoglobin, Platelet counts and other related dependent variables.
		- Data has been sourced from Bellevue University.
		- Feature creation, selection and elimination.
		- Visualization for correlation maxtrix, elbow method plotting for optimum k value, PCA plotting, Silhoutte analysis etc.
		- Standard Scaler and Data preparation.
		- KMeans clustering.
		- Silhoutte score and the WCSS has been used to determine optimal k value.
		- PCA transformation is done after creating the model to visualize how the clusters are formed in a 2 dimensional space of using principal components.
		- Calculating Eigen values and Eigen vectors.
   
	- [Sentiment analyzer using NLTK library - Data prep](https://github.com/riazahmedtgithub/Models/blob/main/Sentiment%20Analyzer%20using%20NLTK.ipynb) and [Sentiment analyzer using NLTK library - model](https://github.com/riazahmedtgithub/Models/blob/main/SentimentAnalysis%20Model.ipynb): Predicting the sentiment of the movie reviews using NLTK libraries,
 		- Data has been sourced from Bellevue University.
		- TextBlob and Vader methods.
		- Tokenized the words using word_tokenize.
		- Removed the punctuation using transalation.
		- Removed the by using stopwords.words('english').
		- Used the PorterStemmer to get the stem of the words.
		- TFIDF Vectorization.
		- Building logistic and SVM model.
		- Model metrics comparison and final model recommendation.

