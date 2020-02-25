import streamlit as st 
import joblib,os
import spacy
import pandas as pd
nlp = spacy.load('en')
import matplotlib.pyplot as plt 
import matplotlib
import base64
matplotlib.use("Agg")
from PIL import Image
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()



def stop_word(vect_text):
	factory2 = StopWordRemoverFactory()
	stopword = factory2.create_stop_word_remover()
	vect_text = stopword.remove(vect_text)
	return vect_text

def stemming(text):
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	task = stemmer.stem(text)
	return task


def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Get the Keys
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key



def main():
	"""News Classifier"""
	st.title("Prototipe ACS")
	#st.subheader("Klasifikasi Deskripsi Lapangan Usaha dengan KBLI")
	html_temp = """
	<div style="background-color:orange;padding:10px">
	<h1 style="color:white;text-align:center;">Koding Deskripsi Lapangan Usaha dengan KBLI </h1>
	</div>

	"""
	
	st.markdown(html_temp,unsafe_allow_html=True)

	st.sidebar.subheader("Pilihan")
	activity = ['Prediksi','NLP']
	choice = st.sidebar.selectbox(" ",activity)
	
	

	if choice == 'Prediksi':
		st.info("One Processing")
		all_ml_models = ["LR","RFOREST","NB"]
		model_choice = st.selectbox("Pilih Model",all_ml_models)

		news_text = st.text_area("Deskripsi Lapangan Usaha")
		prediction_labels = {'90002 Aktivitas Pekerja Seni ': 90002,'49424 Angkutan Ojek Motor': 49424,'71101 Aktivitas Arsitektur': 71101,'93192 Olahragawan, Juri Dan Wasit Profesional': 93192,'56290 Penyedia Makanan Lainya': 56290,'01270 Pertanian Tanaman Untuk Bahan Minuman': 1270,'10130 Industri Pengolahan dan Pengawetan Produk daging dan daging unggas':10130,'10110 Rumah Potong dan Pengepakan Daging Bukan Unggas':10110,'11040 Industri Makanan Ringan':11040,'10772 Industri bumbu masak dan penyedap masakan':10772,'61200 Aktivitas Telekomunikasi Tanpa Kabel':61200,'03225 Budidaya Ikan hias Air Tawar':3225,'03226 Pembenihan Ikan Air Tawar':3226}
		
		if st.button("OneClassify"):
			
			vect_text = [news_text]
			stpwrd = stop_word(str(vect_text))
			stmwrd = stemming(stpwrd)
			vect_text = [stmwrd]
			st.text("Preposesing Text:\n{}".format(vect_text))
			if model_choice == 'LR':
				predictor = load_prediction_models("models/modelLR.pkl")
				prediction = predictor.predict(vect_text)
				result = int(prediction[0])
				#st.write(result)
			elif model_choice == 'RFOREST':
				predictor = load_prediction_models("models/modelRF.pkl")
				prediction = predictor.predict(vect_text)
				result = int(prediction[0])
				# st.write(prediction)
			elif model_choice == 'NB':
				predictor = load_prediction_models("models/modelNB.pkl")
				prediction = predictor.predict(vect_text)
				result = int(prediction[0])
				# st.write(prediction)
			
			final_result = get_key(result,prediction_labels)
			st.success("Hasil Kode KBLI: {}".format(final_result))
		st.info("Batch Processing")
		data = st.file_uploader("Unggah Data", type=["csv", "txt"])
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())
			X = df.iloc[:,0]
			aa = stop_word(str(X))
			#st.text('{}'.format(X))
			if st.button("BatchClassiy"):
			
				predictor = load_prediction_models("models/modelLR.pkl")
				prediction = predictor.predict(X)
				result = prediction
				#st.write(result)
				#df['kode'] = result
				data = {'Deskripsi' : X, 'KBLI' : result}
				newDF = pd.DataFrame(data)
				st.dataframe(newDF.head())
				csv = newDF.to_csv(index=False)
				b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
				href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
				st.markdown(href, unsafe_allow_html=True)

	if choice == 'NLP':
		st.info("Natural Language Processing of Text")
		raw_text = st.text_area("Deskripsi Lapangan Usaha","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analisis"):
			st.info("Original Text::\n{}".format(raw_text))

			docx = nlp(raw_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)

		if st.button("Tabulasi"):
			docx = nlp(raw_text)
			c_tokens = [token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx ]
			c_pos = [token.pos_ for token in docx ]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)


		if st.checkbox("WordCloud"):
			c_text = raw_text
			wordcloud = WordCloud().generate(c_text)
			plt.imshow(wordcloud,interpolation='bilinear')
			plt.axis("off")
			st.pyplot()



	st.sidebar.subheader("About")

if __name__ == '__main__':
	main()

