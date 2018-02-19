import argparse
import math
import json
from os.path import join,dirname
from watson_developer_cloud import ToneAnalyzerV3
from pythonosc import dispatcher
from pythonosc import osc_server
import numpy as np
import time
import pprint
import spotipy
import spotipy.util as util
import sys
import csv
import matplotlib.pyplot as plt


def clearp():
	global token
	if token:
		sp = spotipy.Spotify(auth=token)
		sp.trace = False
		x=sp.user_playlist_tracks('tejasr_42', playlist_id='0styVteGSwoQo01xNT7Ea6')
		for i in x['items']:
			sp.user_playlist_remove_specific_occurrences_of_tracks('tejasr_42', '0styVteGSwoQo01xNT7Ea6', [{'uri':i['track']['uri'],'positions':[0]}])

def add_song(ans):
	global token
	if token:
		sp = spotipy.Spotify(auth=token)
		sp.trace = False
		results = sp.user_playlist_add_tracks('tejasr_42','0styVteGSwoQo01xNT7Ea6', [ans])
		#print(results)
	else:
		print("Can't get token")

tone_analyzer = ToneAnalyzerV3(
		username="56df1500-a0af-4cfc-be9c-e6db36e716a3",
		password="TQyCOx6MUq2H",
		version='2016-05-19')
goals=[[0.3,0.1,0.3,0.7,0.4,0.0,0.0,0.0,0.3,0.2],[0.2,0.8,0.3,0.3,0.3,0.0,0.0,0.1,0.6,0.2],[0.4,0.3,0.5,0.8,0.2,0.0,0.0,0.1,0.6,0.5],[0.6,0.6,0.3,0.2,0.7,0.4,0.0,0.4,0.0,0.2],[0.6,0.6,0.5,0.6,0.6,0.6,0.5,0.4,0.6,0.4]]
moods=["Calm","Excited","Empathetic","Attentive","Complete"]
save=0.0
waves=[0.0,0.0,0.0,0.0,0.0]
count=0
weights=np.random.rand(10,3)*3
weight_delt=np.random.rand(10,3)
pdiff=np.random.rand(10,1)
steps=0
progress=[]

def init():
	tone_analyzer = ToneAnalyzerV3(
		username="56df1500-a0af-4cfc-be9c-e6db36e716a3",
		password="TQyCOx6MUq2H",
		version='2016-05-19')
	goals=[[0.3,0.1,0.3,0.7,0.4,0.0,0.0,0.0,0.3,0.2],[0.2,0.8,0.3,0.3,0.3,0.0,0.0,0.1,0.6,0.2],[0.4,0.3,0.5,0.8,0.2,0.0,0.0,0.1,0.6,0.5],[0.6,0.6,0.3,0.2,0.7,0.4,0.0,0.4,0.0,0.2],[0.6,0.6,0.5,0.6,0.6,0.6,0.5,0.4,0.6,0.4]]
	moods=["Calm","Excited","Empathetic","Attentive","Pseudo-intoxicated"]
	save=0.0
	waves=[0.0,0.0,0.0,0.0,0.0]
	count=0
	weights=np.random.rand(10,3)
	weight_delt=np.random.rand(10,3)*0.1
	pdiff=np.random.rand(10,1)
	steps=0
	progress=[]

def delt(a,b):
	return (a-b)

def dist(x):
	return np.linalg.norm(x)


def alpha_handler(unused_addr, args, ch1, ch2, ch3, ch4):
	waves[0]+=ch1

	
def beta_handler(unused_addr, args, ch1, ch2, ch3, ch4):
	waves[1]+=ch1
	
def delta_handler(unused_addr, args, ch1, ch2, ch3, ch4):
	waves[2]+=ch1

def gamma_handler(unused_addr, args, ch1, ch2, ch3, ch4):
	waves[3]+=ch1
	


def theta_handler(unused_addr, args, ch1, ch2, ch3, ch4):
	global waves
	global count
	global save
	waves[4]+=ch1
	count+=1
	if (time.time()-save)>5:
		server.shutdown()
		fetch()
		print(ch1)
	else:
		pass

def find_nearest(value):
	with open('dinner_track.csv','r') as userFile:
		arr = csv.DictReader(userFile)
		mindist=100
		for elem in arr:
			arr=[float(elem["valence"]),float(elem["speechiness"]),float(elem["energy"])]
			
			x=dist(delt(np.array(arr),value))
			
			if x<mindist:
				mindist = x
				index = elem["id"]
				name=elem["name"]
		print(name)
		return index,name

def sigmoid(m):
	return (1/(1+np.exp(-0.02*m)))

def fetch():
	global waves
	global count
	global save
	global observation
	global pdiff
	global weights
	global weight_delt
	global goals
	global steps
	global progress
	global token
	global em
	steps+=1
	x=input("Start typing\n")
	y=tone_analyzer.tone(x,content_type='text/plain',sentences=False,tones=['emotion'])
	observation=np.array(waves)
	observation/=count
	t=[]
	for i in range(5):
		t.append(y["document_tone"]["tone_categories"][0]["tones"][i]['score'])
	t=np.array(t)
	print(observation)
	print(t)
	observation=np.concatenate([observation,t])
	save=time.time()
	count=0
	waves=[0,0,0,0,0]
	diff=delt(np.array(goals[em]),observation)
	print(observation)
	prog=dist(pdiff)-dist(diff)
	pdiff=diff
	weight_delt=prog*(weight_delt)+(np.random.rand(10,3)*3)
	weights+=weight_delt
	ans=sigmoid(diff.T.dot(weights))
	song,name=find_nearest(np.array(ans))
	add_song(song)
	msg="Added song '"+name+"' to your playlist. Skip to next song."
	print(msg)
	if token:
		sp = spotipy.Spotify(auth=token)
		sp.trace = False
		sp.next_track(device_id='66140894ef53bea6ac91570539be2eda3a9c9956')
	progress.append(observation)
	if dist(diff)<0.8:
		np.save("weights.npy",weights)
		print("Hope you enjoyed the ride")
		progress=np.array(progress)
		plt.plot(progress[:,4])
		return weights
	else:
		server.serve_forever()

if __name__ == "__main__":
	scope = 'playlist-modify-public streaming user-read-playback-state user-modify-playback-state user-read-private'
	token=util.prompt_for_user_token('tejasr_42',scope,client_id='a3d030ec91c0423bafae959d4d3be070',client_secret='700b33711eed40aeabd5640a944f1e12',redirect_uri='http://localhost:8000')
	clearp()
	print("Sensing your zen...")
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", default="10.19.189.5",help="The ip to listen on")
	parser.add_argument("--port",
						type=int,
						default=5000,
						help="The port to listen on")
	args = parser.parse_args()
	em=int(input("Where do you want to be?: \n 0: Calm \n 1: Excited \n 2: Empathetic \n 3: Attentive \n 4: Complete\n"))
	dispatcher = dispatcher.Dispatcher()
	dispatcher.map("/debug", print)
	#dispatcher.map("/muse/eeg", eeg_handler, "EEG")
	dispatcher.map("/muse/elements/alpha_relative", alpha_handler, "Alpha")
	dispatcher.map("/muse/elements/beta_relative", beta_handler, "Beta")
	dispatcher.map("/muse/elements/gamma_relative", gamma_handler, "Gamma")
	dispatcher.map("/muse/elements/delta_relative", delta_handler, "Delta")
	dispatcher.map("/muse/elements/theta_relative", theta_handler, "Theta")
	server = osc_server.ThreadingOSCUDPServer(
		(args.ip, args.port), dispatcher)
	print("Serving on {}".format(server.server_address))
	save = time.time()
	
	server.serve_forever()

	
	