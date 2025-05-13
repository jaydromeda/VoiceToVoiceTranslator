import torch
from TTS.api import TTS
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc , os, sys, argparse, requests
from pathlib import Path



"""#GOOGLE SPEECH TO TEXT
from google.cloud import speech


def speech_to_text(
    config: speech.RecognitionConfig,
    audio: speech.RecognitionAudio,
) -> speech.RecognizeResponse:
    client = speech.SpeechClient()

    # Synchronous speech recognition request
    response = client.recognize(config=config, audio=audio)

    return response


def print_response(response: speech.RecognizeResponse):
    for result in response.results:
        print_result(result)


def print_result(result: speech.SpeechRecognitionResult):
    best_alternative = result.alternatives[0]
    print("-" * 80)
    print(f"language_code: {result.language_code}")
    print(f"transcript:    {best_alternative.transcript}")
    print(f"confidence:    {best_alternative.confidence:.0%}")
    
# Load local audio file as bytes
with open("./output.wav", "rb") as audio_file:
    audio_content = audio_file.read()

# Configure the recognition settings
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=22050,  # Match your TTS output sample rate if different
    language_code="en-US",
)

audio = speech.RecognitionAudio(content=audio_content)

response = speech_to_text(config, audio)
print_response(response)
# Load local audio file as bytes
with open("./output.wav", "rb") as audio_file:
    audio_content = audio_file.read()

# Configure the recognition settings
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=22050,  # Match your TTS output sample rate if different
    language_code="en-US",
)

audio = speech.RecognitionAudio(content=audio_content)

response = speech_to_text(config, audio)
print_response(response)

"""


import speech_recognition as sr

def recognize_local_audio(file_path, rvc_file):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        text1 = recognizer.recognize_google(audio_data)
        print("Transcript:", text1)
    except sr.UnknownValueError:
        text1 = "Could not understand original audio."
    except sr.RequestError as e:
        text1 = f"API error (original): {e}"

    try:
        with sr.AudioFile(rvc_file) as source:
            audio_data = recognizer.record(source)
        text2 = recognizer.recognize_google(audio_data)
        print("RVC Transcript:", text2)
    except sr.UnknownValueError:
        text2 = "Could not understand RVC audio."
    except sr.RequestError as e:
        text2 = f"API error (RVC): {e}"

    return text1, text2


parser = argparse.ArgumentParser(
	prog='XTTS-RVC-UI',
	description='Gradio UI for XTTSv2 and RVC'
)

parser.add_argument('-s', '--silent', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

if args.silent: 
	print('Enabling silent mode.')
	sys.stdout = open(os.devnull, 'w')

def download_models():
	rvc_files = ['hubert_base.pt', 'rmvpe.pt']

	for file in rvc_files: 
		if(not os.path.isfile(f'./models/{file}')):
			print(f'Downloading{file}')
			r = requests.get(f'https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/{file}')
			with open(f'./models/{file}', 'wb') as f:
					f.write(r.content)

	xtts_files = ['vocab.json', 'config.json', 'dvae.path', 'mel_stats.pth', 'model.pth']

	for file in xtts_files:
		if(not os.path.isfile(f'./models/xtts/{file}')):
			print(f'Downloading {file}')
			r = requests.get(f'https://huggingface.co/coqui/XTTS-v2/resolve/v2.0.2/{file}')
			with open(f'./models/xtts/{file}', 'wb') as f:
				f.write(r.content)
				

[Path(_dir).mkdir(parents=True, exist_ok=True) for _dir in ['./models/xtts', './voices', './rvcs']]

download_models()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device: " + device) 

config = Config(device, device != 'cpu')
hubert_model = load_hubert(device, config.is_half, "./models/hubert_base.pt")
tts = TTS(model_path="./models/xtts", config_path='./models/xtts/config.json').to(device)
voices = []
rvcs = []
langs = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]

def get_rvc_voices():
	global voices 
	voices = os.listdir("./voices")
	global rvcs
	rvcs = list(filter(lambda x:x.endswith(".pth"), os.listdir("./rvcs")))
	return [rvcs, voices]

def runtts(rvc, voice, text, pitch_change, index_rate, language): 
    audio = tts.tts_to_file(text=text, speaker_wav="./voices/" + voice, language=language, file_path="./output.wav")
    voice_change(rvc, pitch_change, index_rate)
    return ["./output.wav" , "./outputrvc.wav"]

def main():
	get_rvc_voices()
	print(rvcs)
	print(voices)
	with gr.Blocks(title='TTS RVC UI') as interface:
		with gr.Row():
			gr.Markdown("""
				#XTTS RVC UI
			""")
		with gr.Row(): 
			with gr.Column():
				lang_dropdown = gr.Dropdown(choices=langs, value=langs[0], label='Language')
				rvc_dropdown = gr.Dropdown(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else '', label='RVC model') 
				voice_dropdown = gr.Dropdown(choices=voices, value=voices[0] if len(voices) > 0 else '', label='Voice sample')
				refresh_button = gr.Button(value='Refresh')
				text_input = gr.Textbox(placeholder="Write here...")
				submit_button = gr.Button(value='Submit')
				transcribe_button = gr.Button(value="Transcribe Output")
				transcript_output_1 = gr.Textbox(label="Transcript 1", interactive=False)
				transcript_output_2 = gr.Textbox(label="Transcript 2 (RVC)", interactive=False)

				hidden_path_1 = gr.Textbox(value="./output.wav", visible=False)
				hidden_path_2 = gr.Textbox(value="./outputrvc.wav", visible=False)

				with gr.Row():
					pitch_slider = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch")
					index_rate_slider = gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05, label="Index Rate")
			with gr.Column():        
				audio_output = gr.Audio(label="TTS result", type="filepath", interactive=False)
				rvc_audio_output = gr.Audio(label="RVC result", type="filepath", interactive=False)

		submit_button.click(inputs=[rvc_dropdown, voice_dropdown, text_input, pitch_slider, index_rate_slider, lang_dropdown], outputs=[audio_output, rvc_audio_output], fn=runtts)
		transcribe_button.click(
    		fn=recognize_local_audio,
    		inputs=[hidden_path_1, hidden_path_2],
    		outputs=[transcript_output_1, transcript_output_2]
			)


		def refresh_dropdowns():
			get_rvc_voices()
			print('Refreshed voice and RVC list!')
			return [gr.update(choices=rvcs, value=rvcs[0] if len(rvcs) > 0 else ''),  gr.update(choices=voices, value=voices[0] if len(voices) > 0 else '')] 

		refresh_button.click(fn=refresh_dropdowns, outputs=[rvc_dropdown, voice_dropdown])

	interface.launch(server_name="127.0.0.1", server_port=5000, quiet=True)



# delete later

class RVC_Data:
	def __init__(self):
		self.current_model = {}
		self.cpt = {}
		self.version = {}
		self.net_g = {} 
		self.tgt_sr = {}
		self.vc = {} 

	def load_cpt(self, modelname, rvc_model_path):
		if self.current_model != modelname:
				print("Loading new model")
				del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc
				self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, config.is_half, config, rvc_model_path)
				self.current_model = modelname

rvc_data = RVC_Data()

def voice_change(rvc, pitch_change, index_rate):
	modelname = os.path.splitext(rvc)[0]
	print("Using RVC model: "+ modelname)
	rvc_model_path = "./rvcs/" + rvc  
	rvc_index_path = "./rvcs/" + modelname + ".index" if os.path.isfile("./rvcs/" + modelname + ".index") and index_rate != 0 else ""

	if rvc_index_path != "" :
		print("Index file found!")

	#load_cpt(modelname, rvc_model_path)
	#cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)
	rvc_data.load_cpt(modelname, rvc_model_path)
	
	rvc_infer(
		index_path=rvc_index_path, 
		index_rate=index_rate, 
		input_path="./output.wav", 
		output_path="./outputrvc.wav", 
		pitch_change=pitch_change, 
		f0_method="rmvpe", 
		cpt=rvc_data.cpt, 
		version=rvc_data.version, 
		net_g=rvc_data.net_g, 
		filter_radius=3, 
		tgt_sr=rvc_data.tgt_sr, 
		rms_mix_rate=0.25, 
		protect=0, 
		crepe_hop_length=0, 
		vc=rvc_data.vc, 
		hubert_model=hubert_model
	)
	gc.collect()
    
if __name__ == "__main__":
    main()
