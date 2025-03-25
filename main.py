from flask import Flask, render_template, request, jsonify, send_file
import os
from transformers import pipeline

from backend import Speaker_Diarisation
from user_management import extract_unames, check_existing

# import soundfile as sf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/uploads'
# Create a directory to save audio files if it doesn't exist
if not os.path.exists('audio_files'):
    os.makedirs('audio_files')

from backend import Speaker_identification


@app.route('/home', methods=['GET', 'POST'])
def home():
    uname_list = extract_unames()

    if request.method == 'POST':
        name = request.form.get('uname')
        # Get the uploaded file safely
        audio_file = request.files.get('audio')

        print("Received Name:", name)
        print("Received File:", audio_file.filename)
        complete_audio_file_path = (f'/Users/jaiharishsatheshkumar/PycharmProjects/'
                                    f'lastPhase1/user_wav_temp{audio_file.filename}')
        audio_file.save(complete_audio_file_path)

        embed_upload_check = Speaker_identification().store_user_voice(complete_audio_file_path, name)
        if embed_upload_check:
            return render_template('home.html', uname_list=uname_list, emb_stat=embed_upload_check)

    return render_template('home.html', uname_list=uname_list, emb_stat='')


@app.route('/')
def index():
    uname_list = extract_unames()
    return render_template('index.html', uname_list=uname_list)


# from pydub import AudioSegment

@app.route('/upload', methods=['POST'])
def upload():
    audio_file = request.files['audio']
    client_timestamp = request.form.get('timestamp')
    print(audio_file)

    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(temp_file_path)

    # Convert to PCM WAV format using pydub
    """#'''try:
		# Load the audio file, specifying format if necessary
		audio = AudioSegment.from_file(temp_file_path, format='wav')  # Specify the format if it's WAV

		# Convert to mono and set sample rate to 16 kHz
		audio = audio.set_channels(1).set_frame_rate(16000)

		# Save the final output with the original filename
		final_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
		audio.export(final_file_path, format="wav")

		# Optionally, you can log the successful conversion
		print(f"Successfully converted and saved: {final_file_path}")
	except Exception as e:
		print(f"Error processing audio: {str(e)}")  # Log the error for debugging
		return f"Error processing audio: {str(e)}", 500
	finally:
		# Clean up the temporary file
		if os.path.exists(temp_file_path):
			os.remove(temp_file_path)
"""
    return '', 204  # No content response, as we update the file list on the client


@app.route('/fetch_files', methods=['POST'])
def fetch_files():
    files = []
    for file_name in os.listdir(app.config['UPLOAD_FOLDER']):
        files.append(file_name)
    return jsonify(files)


curr_file = ""


@app.route('/process_file/<filename>', methods=['POST'])
def process_file(filename):
    # global curr_file
    # Perform the action you need (e.g., open, read, or process the file)
    file_path = f'/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/uploads/{filename}'

    try:
        """#curr_file = filename
        # Send back a response with the result of the file processing 
        #return jsonify({"file": filename, "content": send_file(file_path, mimetype="audio/wav")})"""
        return send_file(file_path, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"file": f"Failed to process file '{filename}': {str(e)}"}), 500


transcript_content = ''


def concatenate(lst):
    s = ''
    for i in lst:
        s += i + '\n'
    return s


@app.route('/summarise', methods=['POST'])
def summarise():
    # Example logic to generate summary
    print(concatenate(transcript_content))

    summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum", do_sample=True, temperature=1.8)
    summary_content = summarizer(concatenate(transcript_content), max_length=50)
    return jsonify(summary=summary_content[0]['summary_text'])


@app.route('/generate_transcript/<filename>', methods=['POST'])
def generate_transcript(filename):
    global transcript_content
    file_path = os.path.join('/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/uploads/', filename)
    if not file_path:
        return jsonify(transcript='file not selected')

    transcript_content = Speaker_Diarisation(file_path).conversation_with_speaker_id()

    print(transcript_content)
    return jsonify(transcript=transcript_content)


if __name__ == '__main__':
    app.run(debug=True, port=3000)
