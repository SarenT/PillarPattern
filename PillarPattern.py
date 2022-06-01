#!/usr/bin/env python3
import sys
import numpy as np
import itertools
from pysvg.filter import *
from pysvg.gradient import *
from pysvg.linking import *
from pysvg.script import *
from pysvg.shape import *
from pysvg.structure import *
from pysvg.style import *
from pysvg.text import *
from pysvg.builders import ShapeBuilder, StyleBuilder
from scipy import stats
from scipy.stats import uniform, laplace
from scipy.stats import norm as scinorm
import random

import math

import time

from multiprocessing import Process, Queue
import queue


disMethods = ["gauss", "laplace"]

canvasSize = [120, 120] # width and height
radius = [4.5, 4.5] # in x and y
margins = [4.5, 4.5] # x and y margins around each pillar


freq = 20 # mm frequency of stepwise or sinusoidal pore size change over distance
minPoreSize = 6 # mm
maxPoreSize = 12 # mm
poreSizeRange = [minPoreSize, maxPoreSize]

addIndicators = False # attraction points in branching design
# All units are in mm, for μm scale down 1:1000 in Inkscape
unit = 'mm' # SVG file unit

shape = 'circle'

styleFill = "#000000"

defaultArgs = []

csvExport = True

subDir = '' # Subdirectory
mainDir = '' # Root Directory
parentDir = mainDir + subDir # Directory where files are saved
filetype = 'svg'

alternativeSpace = 50
aroundLimit = 170
aroundSpace = 50
aroundSpaceFill = 50

disMethod = "gauss"

shapeB = ShapeBuilder()
# styleB = StyleBuilder()
# styleB.setStrokeWidth(2)
# styleB.setStroke("green")

#http://www.southampton.ac.uk/~fangohr/training/python/snippets/lecture09/mexhat-numpy.py
def mexhat_np(t, sigma=1):
    """Computes Mexican hat shape using numpy, see
    http://en.wikipedia.org/wiki/Mexican_hat_wavelet for
    equation (13 Dec 2011)"""
    c = 2. / math.sqrt(3 * sigma) * math.pi ** 0.25
    return c * (1 - t ** 2 / sigma ** 2) * \
        np.exp(-t ** 2 / (2 * sigma ** 2))

def getAttractionForce(forceFunc, strength, r):
	
	if forceFunc == 0:# or forceFunc == "gaussian" or forceFunc == "g":
		if r < strength*1.2:
			#f = strength / (r**2)
			f = stats.norm.pdf(r, scale = strength*2)
		else:
			f = 0
	elif forceFunc == 1:# or forceFunc == "trigonometric" or forceFunc == "t":
		if r > strength:
			f = 0
		else:
			f = (np.cos(np.pi * r/(1 * strength)) + 1)/2
	elif forceFunc == 2:# or forceFunc == "mexicanhat" or forceFunc == "m":
		sigma = strength/2
		t = np.arange(-10, 10, 0.01)
		phi = -1 * mexhat_np(t/2 + 3*sigma/2, sigma)
		f = -1 * mexhat_np(r/2 + 3*sigma/2, sigma)
		f /= max(phi)
		
	return f

def valsFormatted(vals):
	a = [None] * len(vals)
	for i in range(len(vals)):
		if isinstance(vals[i], bool):
			a[i] = str(int(vals[i]))
		elif isinstance(vals[i], int): 
			a[i] = str(vals[i])
		elif isinstance(vals[i], str):
			a[i] = vals[i]
		elif isinstance(vals[i], float):
			a[i] = str(vals[i])
			
	return('|'.join(a))

def prepareFileNames():
	global suffix, suffixAround, svgFileRegular, svgFileDisarranged, svgFilePolka, svgFileShiftedPairs
	global svgFileRadSinusoidalLongVar, svgFileRadSinusoidalLongLatVar
	global svgFileRadStepLongVar, svgFileRadStepLongLatVar
	global svgFileAxialSinusoidalLongVar, svgFileAxialSinusoidalLongLatVar
	global svgFileAxialStepLongVar, svgFileAxialStepLongLatVar
	global svgFileRadBranching, svgFileAxBranching, svgFileRadPeriodicPerp
	 
	suffix =  str(radius[0] * 2) + '-' + str(radius[1] * 2) + 'mm-pillar' + '_' + str(margins[0] * 2) + 'mm-gap'
	suffixAround =  suffix + '_Around' # Around - corner center circle empty, corner mid circle full of pillars
	fileEnding = "{}x{}_".format(canvasSize[0], canvasSize[1]) + shape + "_" + suffix + '.' + filetype
	fileEndingAround = shape[:4] + "_" + suffixAround + '.' + filetype
	
	svgFileRegular = parentDir + 'regular_{}_{}_' + fileEnding # A rectangular space filled with pillars
	
	svgFileDisarranged = parentDir + 'disarranged_{}_{}_scale{}_' + fileEnding
	
	#svgFilePairShifted = parentDir + 'pairshifted_' + suffix
	svgFilePolka = parentDir + 'polka_{}_{}_' + fileEnding
	
	#svgFileShiftedPillars = parentDir + 'shiftedpillars_' + fileEnding
	
	svgFileShiftedPairs = parentDir + 'shiftedpairs_{}_{}_' + fileEnding
	
	svgFileRadSinusoidalLongVar = parentDir + 'rad-sin_Long_freq{}_' + fileEnding
	svgFileRadSinusoidalLongLatVar  = parentDir + 'rad-sin_LongLat_freq{}_' + fileEnding
	
	svgFileRadStepLongVar = parentDir + 'rad-stp_Long_freq{}_' + fileEnding
	svgFileRadStepLongLatVar  = parentDir + 'rad-stp_LongLat_freq{}_' + fileEnding
	
	svgFileAxialSinusoidalLongVar = parentDir + 'ax-sin_Long_freq{}_' + fileEnding
	svgFileAxialSinusoidalLongLatVar  = parentDir + 'ax-sin_LongLat_freq{}_' + fileEnding
	
	svgFileAxialStepLongVar = parentDir + 'ax-stp_Long_freq{}_' + fileEnding
	svgFileAxialStepLongLatVar  = parentDir + 'ax-stp_LongLatVar_freq{}_' + fileEnding
	
	svgFileRadBranching = parentDir + 'rad-brnch_st{}_sc{}_os{}-{}_{}_{}_nrow{}_{}_spr{}_{}_' + fileEnding
	svgFileAxBranching = parentDir + 'ax-brnch_st{}_sc{}_os{}-{}_{}_{}_nrow{}_{}_' + fileEnding
	 
	svgFileRadPeriodicPerp = parentDir + 'rad-per-perp_ps{}_st{}_{}_{}_co{}_per{}_sc{}_' + fileEnding
	
def putShape(cont, shapeB, shape, p, size, unit, fill = styleFill, stroke = "black", strokewidth = 0, opacity = 1.0, 
			 isPath = True):
	shapeObj = None
	
	styleB = StyleBuilder()
	if stroke is not None:
		styleB.setStroke(stroke)
	if fill is not None:
		styleB.setFilling(fill)
	if strokewidth is not None:
		styleB.setStrokeWidth(strokewidth)
		
	if shape == 'circle':
		if isPath:
			
			factor = 96/25.4
			midPos = [d for d in p]
			startPos = [pos * factor for pos in midPos.copy()]
			startPos[0] += size[0] * factor
			nodeTemplate = "{},{} 0 0 1 {},{} "
			circlePattern = [[-1, 1], [-1, -1], [1, -1], [1, 1]]
			nodes = [nodeTemplate.format(size[0] * factor, size[0] * factor, 
								size[0] * circlePattern[i][0] * factor, size[0] * circlePattern[i][1] * factor) 
			for i in range(len(circlePattern))]
			
			d="M {},{} a {}Z".format(startPos[0], startPos[1], ''.join(nodes))
			shapeObj = Path(d, style = styleB.getStyle())
		else:
			shapeObj = shapeB.createCircle(str(p[0]) + unit, str(p[1]) + unit, str(size[0]) + unit, 
									 strokewidth=strokewidth, stroke=stroke, fill=fill)#, opacity=opacity))
	elif shape == 'rectangle' or shape == 'rect':
		shapeObj = shapeB.createRect(str(p[0] - size[0]) + unit, str(p[1] - size[1]) + unit, str(size[0] * 2) + unit, str(size[1] * 2) + unit, 
								   strokewidth=strokewidth, stroke=stroke, fill=fill)#, opacity=opacity))
	if shapeObj != None:
		shapeObj.setAttribute('style', shapeObj.getAttribute('style') + 'opacity:' + str(opacity))
		cont.addElement(shapeObj)
		
def cart2pol(p):    
    return([np.sqrt(p[0]**2 + p[1]**2), np.arctan2(p[1], p[0])])

def pol2cart(pol):
    return([pol[0] * np.cos(pol[1]), pol[0] * np.sin(pol[1])])

def pointsDistance(p1, p2):
	return norm([p1[0] - p2[0], p1[1] - p2[1]])

def minVec(v1, v2):
	if pointsDistance([0,0], v1) < pointsDistance([0,0], v2):
		return v1
	else:
		return v2

def getClosestPillar(pillars, p, limit):
	pClosest = None
	lastDist = sys.float_info.max
	for pillar in pillars:
		curDist = pointsDistance(pillar, p)
		if curDist < lastDist and curDist < limit:
			pClosest = pillar
			lastDist = curDist
	if pClosest is None:
		return p
	else:
		return pClosest

def norm(p):
	return np.sqrt(np.power(p[0], 2) + np.power(p[1], 2))

def newDefaultDoc():
	global unit, canvasSize
	return newDoc([0, 0], canvasSize, unit)

def getArguments(inputs, defaultArgs):	
	vals = {}
	for inp in inputs:
		if len(defaultArgs) > 0:
			val = inp[1](defaultArgs.pop(0))
		else:
			val = inp[1](input(inp[2]) or inp[3])
		vals[inp[0]] = val
	return vals, defaultArgs

def interactiveCommands(choiceNames, message, allowAll = True, manual = True, allowExit = True, defaultArgs = []):
	nChoices = len(choiceNames)
	choices = [False] * nChoices
	allInd = -1
	exitInd = -1
	if allowAll:
		choiceNames.append("all")
		allInd = len(choiceNames) - 1
	if allowExit:
		choiceNames.append("exit")
		exitInd = len(choiceNames) - 1
	
	#typeNames = ["all", "normal", "around", "around-fill", "around-non-fill", "fill", "alternative"]
	choiceDesc = [choiceName + "[" + str(ind) + "]" for ind, choiceName in enumerate(choiceNames)]
	if manual:
		if len(defaultArgs) > 0:
			command = defaultArgs.pop(0)
		else:
			command = input(message + "\nOptions (" + ', '.join(choiceDesc) + "):\n") # "What type (overall pattern area shape) should to be created?
			
		if command.isdigit():
			commandInd = int(command)
			if (commandInd < nChoices):
				choices[commandInd] = True
			else:
				if commandInd == allInd and allowAll:
					choices = [True] * nChoices
				elif commandInd == exitInd and allowExit:
					choices = -1
				else:
					print("Wrong input, number too high")
					return
		elif command in choiceNames:
			 commandInd = choiceNames.index(command)
		else:
			print("Unknown command")
			return
	else:
		choices = [True] * nChoices
	
	return(choices)

def parseArgs(args):
	global canvasSize, radius, margins, unit, parentDir, freq, poreSizeRange, alternativeSpace
	global aroundLimit, aroundSpace, aroundSpaceFill, shape, defaultArgs, mainDir
	for arg in args:
		splitted = arg.split('=', 2)
		print(splitted)
		if splitted[0] == 'width':
			canvasSize[0] = float(splitted[1])
		elif splitted[0] == 'height':
			canvasSize[1] = float(splitted[1])
		elif splitted[0] == 'mainDir':
			mainDir = splitted[1]
			parentDir = mainDir + subDir
		elif splitted[0] == 'radius':
			radiusSplit = splitted[1].split(';')
			if len(radiusSplit) == 2:
				radius = [float(radiusSplit[0]), float(radiusSplit[1])]
			elif len(radiusSplit) == 1:
				radius = [float(radiusSplit[0]), float(radiusSplit[0])]
			else:
				print("radius not recognized: use either single value e.g. radius=10.2 or double value e.g. radius=10.2;20.4")
		elif splitted[0] == 'unit':
			unit = splitted[1]
		elif splitted[0] == 'margin':
			margins[0] = float(splitted[1])
			margins[1] = margins[0]
		elif splitted[0] == 'xMargin':
			margins[0] = float(splitted[1])
		elif splitted[0] == 'yMargin':
			margins[1] = float(splitted[1])
		elif splitted[0] == 'dir':
			parentDir = splitted[1]
		elif splitted[0] == 'freq':
			freq = float(splitted[1])
		elif splitted[0] == 'minPoreSize':
			poreSizeRange[0] = float(splitted[1])
		elif splitted[0] == 'maxPoreSize':
			poreSizeRange[1] = float(splitted[1])
		elif splitted[0] == 'alternativeSpace':
			alternativeSpace = float(splitted[1])
		elif splitted[0] == 'aroundLimit':
			aroundLimit = float(splitted[1])
		elif splitted[0] == 'aroundSpace':
			aroundSpace = float(splitted[1])
		elif splitted[0] == 'aroundSpaceFill':
			aroundSpaceFill = float(splitted[1])
		elif splitted[0] == 'shape':
			shape = splitted[1]
		elif splitted[0] == 'args':
			defaultArgs = splitted[1].split('|')
			
def inBoundary(p, margins, circOuterLimit = -1):
	global canvasSize, radius
	
	if circOuterLimit > -1:
		normDistance = np.sqrt(np.power(p[0], 2) + np.power(p[1], 2))
		if normDistance > circOuterLimit:
			return False
		else:
			return True
			
	if (p[0] >= (-1 * (radius[0] + margins[0]) * 2) and p[0] <= (canvasSize[0] + (radius[0] * 2))) and \
	(p[1] >= (-1 * (radius[1] + margins[1]) * 2) and p[1] <= (canvasSize[1] + (radius[1] * 2))):
		return True
	else:
		return False

def newDoc(origin = [0, 0], size = [200, 200], unit = "px"):
	svg = Svg(str(origin[0]) + unit, str(origin[1]) + unit, str(size[0]) + unit, str(size[1]) + unit)
	#svg.set_viewBox("0 0 {} {}".format(size[0] * (96/25.4), size[1] * (96/25.4)))
	
	dictSVG = {"xmlns:dc":"http://purl.org/dc/elements/1.1/", "xmlns:cc":"http://creativecommons.org/ns#", 
			   "xmlns:rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#", "xmlns:svg":"http://www.w3.org/2000/svg",
			   "xmlns":"http://www.w3.org/2000/svg", 
			   "xmlns:sodipodi":"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
			   "xmlns:inkscape":"http://www.inkscape.org/namespaces/inkscape"}
	for key in dictSVG.keys():
		svg.setAttribute(key, dictSVG[key])
		#print(key + " " + dictSVG[key])
		
	dict = {"pagecolor":"#808080", "bordercolor":"#666666", "borderopacity":"1", "objecttolerance":"10", 
			"gridtolerance":"10", "guidetolerance":"10", "inkscape:pageopacity":"0", "inkscape:pageshadow":"2", 
			"inkscape:window-width":"1855", "inkscape:window-height":"1056", "showgrid":"false", 
			"inkscape:document-units":unit, "inkscape:zoom":"0.22201482", "inkscape:cx":"1185.4832", 
			"inkscape:cy":"1423.5187", "inkscape:window-x":"65", "inkscape:window-y":"24", 
			"inkscape:window-maximized":"1", "units":unit}
	element = BaseElement("sodipodi:namedview")
	
	for key in dict.keys():
		element.setAttribute(key, dict[key])
		#print(key + " " + dict[key])
		
	
	svg.addElement(element)
	
	return svg

def shiftedPillars(fileName, canvasSize, radius, margins, unit, shapeB, circSpace = -1, circOuterLimit = -1, fillType = "shiftedpillars"):	
	fileName = fileName.format(fillType, shape)
	doc = newDoc([0, 0], canvasSize, unit)
	points = []
	
	if csvExport:
		f = open(fileName + ".csv", "w")
	p_loc = [x for x in radius]
	
	row = 0
	col = 0
	while True:
		#print p_loc[0], p_loc[1]
		if inBoundary(p_loc, margins, circOuterLimit):
			point = p_loc.copy()
			#points.append(point)
			p_loc[0] += (radius[0] + margins[0]) * 2
			col += 1
			put = True
			
			normDistance = norm(point) # Distance from center
			if circSpace > -1:
				#print(normDistance)
				if normDistance < circSpace:	
					put = False
					 
			if circOuterLimit > -1:
				if normDistance > circOuterLimit:
					put = False
					
			if put:
				putShape(doc, shapeB, shape, point, radius, unit)
				if csvExport:
					f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
			
		else:
			#print('New row!'
			row += 1
			col = 0
			p_loc[0] = radius[0] - ((radius[0] + margins[0]) * (row % 2))			
			p_loc[1] += radius[1]*2 + margins[1]*2
			#print p_loc[0], p_loc[1], 0, 0, 0
			if inBoundary(p_loc, margins, circOuterLimit):
				point = p_loc.copy()
				#points.append(point)
				p_loc[0] += (radius[0] + margins[0]) * 2
				put = True
				
				normDistance = norm(point)
				if circSpace > -1:
					#print(normDistance)
					if normDistance < circSpace:	
						put = False
				
				if circOuterLimit > -1:
					if normDistance > circOuterLimit:
						put = False
				if put:
					putShape(doc, shapeB, shape, point, radius, unit)
					if csvExport:
						f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
				
			else:
				break
	
	if csvExport:
		f.close()
		
	doc.save(fileName)
	print("File saved to " + fileName)
    
def polkaDots(fileName, canvasSize, radius, margins, unit, shapeB, circSpace = -1, circOuterLimit = -1, fileType = "normal"):
	fileName = fileName.format(fillType, shape)
	doc = newDoc([0, 0], canvasSize, unit)
	points = []
	if csvExport:
		f = open(fileName + ".csv", "w")
	p_loc = [x for x in radius]
	
	row = 0
	col = 0
	while True:
		#print p_loc[0], p_loc[1]
		if inBoundary(p_loc, margins, circOuterLimit):
			point = p_loc.copy()
			#points.append(point)
			p_loc[0] += (radius[0] + margins[0]) * 2
			col += 1
			put = True
			
			normDistance = norm(point)
			if circSpace > -1:
				#print(normDistance)
				if normDistance < circSpace:	
					put = False
					 
			if circOuterLimit > -1:
				if normDistance > circOuterLimit:
					put = False
					
			if put:
				putShape(doc, shapeB, shape, point, radius, unit)
				if csvExport:
					f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
			
		else:
			#print('New row!'
			row += 1
			col = 0
			p_loc[0] = radius[0] - ((radius[0] + margins[0]) * (row % 2))
			if shape == 'circle':
				p_loc[1] += (radius[1] + margins[1]) * np.tan(np.pi / 3) # 60°
			elif shape == 'rect' or 'rectangle':
				p_loc[1] += (radius[1] + margins[1]) * 2
				
			#print p_loc[0], p_loc[1], 0, 0, 0
			if inBoundary(p_loc, margins, circOuterLimit):
				point = p_loc.copy()
				#points.append(point)
				p_loc[0] += (radius[0] + margins[0]) * 2
				put = True
				
				normDistance = norm(point)
				if circSpace > -1:
					#print(normDistance)
					if normDistance < circSpace:	
						put = False
				
				if circOuterLimit > -1:
					if normDistance > circOuterLimit:
						put = False
				if put:
					putShape(doc, shapeB, shape, point, radius, unit)
					if csvExport:
						f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
				
			else:
				break
	
	if csvExport:
		f.close()
	doc.save(fileName)
	print("File saved to " + fileName)

def shiftedPairs(fileName, canvasSize, radius, margins, unit, shapeB, circSpace = -1, circOuterLimit = -1, fillType = "normal"):
	fileName = fileName.format(fillType, shape)
	doc = newDoc([0, 0], canvasSize, unit)
	points = []
			
	p_loc = [x for x in radius]
	
	margins[0] = margins[0] / 2
	row = 0
	col = 0
	if csvExport:
		f = open(fileName + ".csv", "w")
		
	while True:
		#print p_loc[0], p_loc[1]
		if inBoundary(p_loc, margins, circOuterLimit):
			point = p_loc.copy()
			p_loc[0] += (radius[0] + margins[0]) * 2
			#points.append(point)
			put = True
			normDistance = norm(point)
			if circSpace > -1:
				#print(normDistance)
				if normDistance < circSpace:	
					 put = False
			
			if circOuterLimit > -1:
				if normDistance > circOuterLimit:
					put = False
			
			if put:
				putShape(doc, shapeB, shape, point, radius, unit)
				if csvExport:
					f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
				#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
				#								   stroke='black', fill='white'))
				
				
			
			point = p_loc.copy()
			p_loc[0] += (radius[0] + margins[0]) * 2 * 3
			#points.append(point)
			put = True
			
			normDistance = norm(point)
			if circSpace > -1:
				#print(normDistance)
				if normDistance < circSpace:	
					put = False
					 
			if circOuterLimit > -1:
				if normDistance > circOuterLimit:
					put = False
					
			if put:
				putShape(doc, shapeB, shape, point, radius, unit)
				if csvExport:
					f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
				#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
				#								   stroke='black', fill='white'))
				
			
			#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
			#								   stroke='black', fill='white'))
			
			
			col += 1
		else:
			#print('New row!'
			row += 1
			col = 0
			p_loc[0] = radius[0] + ((radius[0] + margins[0]) * (row % 2)) * 4
			p_loc[1] += (radius[1] + margins[1])
			#print p_loc[0], p_loc[1], 0, 0, 0
			if inBoundary(p_loc, margins, circOuterLimit):
				point = p_loc.copy()
				p_loc[0] += (radius[0] + margins[0]) * 2
				#points.append(point)
				put = True
				
				normDistance = norm(point)
				if circSpace > -1:
					#print(normDistance)
					if normDistance < circSpace:	
						put = False
				if circOuterLimit > -1:
					if normDistance > circOuterLimit:
						put = False
					
				if put:
					putShape(doc, shapeB, shape, point, radius, unit)
					if csvExport:
						f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
					#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
					#								   stroke='black', fill='white'))
				#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
				#								   stroke='black', fill='white'))
				
				point = p_loc.copy()
				p_loc[0] += (radius[0] + margins[0]) * 2 * 3
				
				put = True
				
				normDistance = norm(point)
				if circSpace > -1:
					#print(normDistance)
					if normDistance < circSpace:	
						 put = False
				
				if circOuterLimit > -1:
					if normDistance > circOuterLimit:
						put = False
				if put:
					putShape(doc, shapeB, shape, point, radius, unit)
					if csvExport:
						f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
					#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
					#								   stroke='black', fill='white'))
				#points.append(point)
				#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
				#								   stroke='black', fill='white'))
				
			else:
				break
	
	if csvExport:
		f.close()
		
	doc.save(fileName)
	print("File saved to " + fileName)

def regular(fileName, canvasSize, radius, margins, unit, shapeB, scale = 0.5, circSpace = -1, circOuterLimit = -1, 
			regFill = False, group = 0, fillType = "normal"):
	"""
	Generates a SVG file of regular grid of pillars.
	
	Parameters
	----------
	fileName : str
		Path where to save SVG file
	width : float
		document width
	height : float
		document height
	radius : float
		Radius of all pillars
	margins[0] : float
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	margins[1] : float
		Similar to margins[0] but on y axis.
	unit : str
		Spatial unit of the document
	shapeB : ShapeBuilder
		ShapeBuilder object to generate circles
	scale : float
		not required
	circSpace : float
		Distance limit for fill space. In combinaton with regFill
	circOuterLimit : float
		Outer limit for pillar from the upper left corner. -1 (default) is ignored (no limit).
	regFill : bool
		True=inside (top left corner filled), False=outside (away from top left corner filled)
	Returns
	-------
	float[][]
		A two-dimensional array, where first dimension defines each point and second dimension represents axes (x and y)
	"""
		
	fileName = fileName.format(fillType, shape)
	doc = newDoc([0, 0], canvasSize, unit)
	points = regularGrid(radius, margins, circOuterLimit)
	
	groups = [[G() for y in range(group)] for x in range(group)]
	
	if csvExport:
		f = open(fileName + ".csv", "w")
	#print(regFill)
	for i in range(0, len(points)):
		# Regular Shape
		normDistance = np.sqrt(np.power(points[i][0], 2) + np.power(points[i][1], 2))
		
		if circSpace > -1:			
			#print((normDistance, circSpace))
			if regFill == False and normDistance < circSpace:	
				continue
			elif regFill == True and normDistance >= circSpace:
				continue
			
		if circOuterLimit > -1:
			if normDistance > circOuterLimit:
				continue
		if len(groups) > 0:
			x = canvasSize[0]
			y = canvasSize[1]
			ind = [0, 0]
			for n in range(2):
				ind[n] = np.round(np.multiply(group, np.divide(points[i][n], canvasSize[n])))
				ind[n] = int(min(ind[n], group - 1))
			cont = groups[ind[0]][ind[1]]
		else:
			cont = doc
		putShape(cont, shapeB, shape, points[i], radius, unit)
		
		if csvExport:
			f.write(str(points[i][0]) + "\t" + str(points[i][1]) + "\n")
		#doc.addElement(shapeB.createCircle(str(points[i][0]) + unit, str(points[i][1]) + unit, str(radius) + unit, 0, 
		#								   stroke='black', fill='white'))
		
	if csvExport:
		f.close()
	for gr in groups:
		for g in gr:
			doc.addElement(g)
	doc.save(fileName)
	print("File saved to " + fileName)
	
def narrowingChannels(fileName, canvasSize, radius, margins, unit, shapeB, scale = 0.5, circSpace = -1):
	doc = newDoc([0, 0], canvasSize, unit)
	points = regularGrid(radius, margins)
    
	for i in range(0, len(points)):
		# Regular Shape
		putShape(doc, shapeB, shape, points[i], radius, unit)
		#doc.addElement(shapeB.createCircle(str(points[i][0]) + unit, str(points[i][1]) + unit, str(radius) + unit, 0, stroke='black', fill='white'))
	doc.save(fileName)

def channels(radius, margins):
	x = range(margins[0] * 2, canvasSize[0], (radius[0] + margins[0]) * 2)
	y = range(margins[1] * 2, canvasSize[1], (radius[1] + margins[1]) * 2)
	points = list(itertools.product(x, y))
	return points	

def regularGrid(radius, margins, circOuterLimit = -1):
	global canvasSize
	"""
	Generates a regular grid of pillars.
	
	Parameters
	----------
	radius : float
		Radius (size) of all pillars
	margins : list
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	circOuterLimit : float
		Outer limit for pillar from the upper left corner. -1 (default) is ignored (no limit).
	
	Returns
	-------
	float[][]
		A two-dimensional array, where first dimension defines each point and second dimension represents axes (x and y)
	"""
	#ipdb.set_trace()
	unitX = (radius[0] + margins[0]) * 2
	unitY = (radius[1] + margins[1]) * 2
	
	if(circOuterLimit > -1):
		specWidth = circOuterLimit + radius[0]
		specHeight = circOuterLimit + radius[1]
	else:
		specWidth = canvasSize[0] + radius[0]
		specHeight = canvasSize[1] + radius[1]
	
	
	
	x = np.arange(radius[0], specWidth, unitX)
	y = np.arange(radius[1], specHeight, unitY)
	#x = [float(a)/1000 for a in x]
	#y = [float(b)/1000 for b in y]
	points = list(itertools.product(x, y))
	return points

def disarranged(fileName, canvasSize, radius, margins, unit, shapeB, scale = 0.5, circSpace = -1, circOuterLimit = -1,
				method = "gauss", fillType = "normal"):
	"""
	Generates a SVG file of regular grid of pillars.
	
	Parameters
	----------
	fileName : str
		Path where to save SVG file
	width : float
		document width
	height : float
		document height
	radius : float
		Radius of all pillars
	margins[0] : float
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	margins[1] : float
		Similar to margins[0] but on y axis.
	unit : str
		Spatial unit of the document
	shapeB : ShapeBuilder
		ShapeBuilder object to generate circles
	scale : float
		Scale for random variates from laplace distribution
	circSpace : float
		Distance limit for fill space. In combinaton with regFill
	circOuterLimit : float
		Outer limit for pillar from the upper left corner. -1 (default) is ignored (no limit).
	regFill : bool
		True=inside (top left corner filled), False=outside (away from top left corner filled)
	Returns
	-------
	float[][]
		A two-dimensional array, where first dimension defines each point and second dimension represents axes (x and y)
	"""
	fileName = fileName.format(method, fillType, scale)
	doc = newDoc([0, 0], canvasSize, unit)
	points = regularGrid(radius, margins, circOuterLimit)
	
	if method == "laplace": # Initially this was used
		# Be aware! For randomness, margins[0] is taken.
		r = laplace.rvs(loc = scale, scale = scale, size=len(points))
		angle = uniform.rvs(scale = np.pi * 2, size=len(points))
		xShift = np.cos(angle) * np.abs(r)
		yShift = np.sin(angle) * np.abs(r)
	elif method == "gauss":
		xShift = scinorm.rvs(loc = 0, scale = scale, size = len(points))
		yShift = scinorm.rvs(loc = 0, scale = scale, size = len(points))
		
	if csvExport:
		f = open(fileName + ".csv", "w")
		
	for i in range(0, len(points)):
		#print angle[i], r[i]
		point = (points[i][0] + xShift[i], points[i][1] + yShift[i])
		#points.append(point)
		
		if not inBoundary(point, margins):
			#print("not in bound", point)
			point = [random.random() * dim for dim in canvasSize]
			#print("\t", point)
			
		normDistance = norm(point)
		if circSpace > -1:
			#print(normDistance)
			if normDistance < circSpace:	
				continue
			
		if circOuterLimit > -1:
			if normDistance > circOuterLimit:
				continue
		putShape(doc, shapeB, shape, point, radius, unit)
		if csvExport:
			f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
		#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, 
		#								   stroke='black', fill='white'))
	if csvExport:
		f.close()
		
	doc.save(fileName)
	print("File saved to " + fileName)

def sinusoidal(sizeRange, freq, x):
	"""
	Sinus function to calculate 
	
	Parameters
	----------
	maxDist : float
		Max distance required between 2 points.
	minDist : float
		Min distance required between 2 points.
	freq : float
		Space between narrowest pores
	x : float
		Distance from start
	Returns
	-------
	float
		y or distance between points in longitudinal
	"""
	return((((math.sin(x * 2 * math.pi / freq) + 1) / 2) * (sizeRange[1] - sizeRange[0])) + sizeRange[0])

def calculateAttractionForces(threadID, q, result):
	print("Thread is starting: %d" % threadID)
	while True:
		try:
			task = q.get_nowait()
		except queue.Empty:
			break
		else:	
			i = task[0]
			attractionPoints = task[1]
			points = task[2]
			forceFunc = task[3]
			strength = task[4]
			strengthFactor = task[5]
			r = task[5]
			#fs = task[6]
			if i % 50 == 0:
				print("F: ", i, "/", len(attractionPoints), threadID)
			else:
				print(threadID, end = "")
			aPoint = attractionPoints[i]
			strength = aPoint[2]
			forces = np.zeros((len(points), 2)) # Forces for all pillars
			#ipdb.set_trace()
			for n in range(len(points)):
				#ipdb.set_trace()
				point = points[n]
				r = pointsDistance(aPoint, point)
				
				f = getAttractionForce(forceFunc, strength, r)
	
				#fs.append(f)				
				vec = aPoint[0:2] - point
				try:
					normVec = vec / norm(vec)
				except:
					print(vec)
				forces[n] = minVec(strengthFactor * strength * f * normVec, vec)
				
			
			#print(forces, result)
			result.put(forces)
			
			#q.task_done()
	print("Thread is finishing: %d" % threadID)
	
	return True
		
def branchingRadial(fileName, canvasSize, radius, margins, unit, shapeB, center, forceFunc = 0, repeatRow = 1,\
					shiftRows = 1, centerOffset = 0, startNumber = 6, scaleDown = 1.0, k = 1.0, strengthFactor = 30,
					forceFuncTitle = "gaussian", fromCorner = 1, centerOffsetRm = 0, spread = 6, closestPillar = False):
	global addIndicators, csvExport
	"""
	Generates a SVG file of regular grid of pillars.
	
	Parameters
	----------
	fileName : str
		Path where to save SVG file
	canvasSize : np array
		document width and height
	radius : float
		Radius of all pillars
	margin : np array
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	unit : str
		Spatial unit of the document
	shapeB : ShapeBuilder
		ShapeBuilder object to generate circles
	center : np array
		center position
	centerOffset : float
		starting from center
	startNumber : int
		number of attraction centers in the first row
	scaleDown : float
		factor by to increase number of attraction points at each row
	Returns
	-------
	float[][]
		A two-dimensional array, where first dimension defines each point and second dimension represents axes (x and y)
	"""
	
	fileName = fileName.format(startNumber, scaleDown, centerOffset, centerOffsetRm, forceFuncTitle[:4], \
							strengthFactor, repeatRow, "shft" if bool(shiftRows) else "nsft", spread, closestPillar)
	
	doc = newDoc([0, 0], canvasSize, unit)
	points = np.asarray(regularGrid(radius, margins))
	#print(points.shape)
	attractionPoints = np.empty([0, 3])
	
	rowR = centerOffset
	nAPointsInRow = startNumber
	
	rowRepeatCounter = repeatRow
	prevAlpha = -1
	diag = (canvasSize[0] * (2**0.5))
	if not fromCorner:
		diag /= 2
	
	circum = 2 * np.pi * rowR
	strength = (circum / nAPointsInRow)/2
	
	dAlpha = 2 * np.pi / startNumber
	
	maxRadius = max(radius)
	
	while rowR < diag:
		if prevAlpha != 0 or shiftRows == 0:
			alpha = 0
		else:
			alpha = dAlpha
			
		prevAlpha = alpha
		#print(dAlpha)
		while (alpha + 2*np.pi/1000) < (2 * np.pi):
			polarCoord = [rowR, alpha]
			
			cartCoord = np.asarray(pol2cart(polarCoord))
			if inBoundary(cartCoord, [strength, strength], max(canvasSize) + strength):
				if closestPillar:
					cartCoord = getClosestPillar(points, cartCoord, radius[0] + margins[0])
					
				if not bool(fromCorner):
					cartCoord = cartCoord + center
				cartCoord = np.append(cartCoord, strength)
				if addIndicators:
					putShape(doc, shapeB, shape, cartCoord, [strength] * 2, unit, fill = "#00FF00", opacity = 0.5)
					putShape(doc, shapeB, shape, cartCoord, radius, unit, fill = "red")
				
				attractionPoints = np.vstack((attractionPoints, cartCoord))
			alpha += dAlpha

		print(rowR, "/", diag, (cartCoord[0]**2 + cartCoord[1]**2)**0.5, strength, rowRepeatCounter, len(attractionPoints))
		
		if rowRepeatCounter == 1:
			rowRepeatCounter = repeatRow
#			nAPointsInRow = round(nAPointsInRow * scaleDown)
			
			rowR += strength + strength / scaleDown
			strength /= scaleDown
			circum = 2 * np.pi * rowR
		else:
			rowRepeatCounter -= 1
			rowR += strength + strength
		
		circum = 2 * np.pi * rowR
		nAPointsInRow = (circum / strength) / 2
		#strength = (circum / nAPointsInRow)/2
		
		dAlpha = 2 * np.pi / nAPointsInRow
		
		if strength < maxRadius * 8:
			break
	
	allForces = np.zeros((len(points), 2))
	#fs = [] # len(attractionPoints) * len(points)/pillars for summaries
	#ipdb.set_trace()

	results = [None] * len(attractionPoints)
	q = Queue(len(attractionPoints))
	qResults = Queue(len(attractionPoints))
	num_theads = min(6, len(attractionPoints))
	
	for i in range(len(attractionPoints)):
		q.put((i, attractionPoints, points, forceFunc, strength, strengthFactor))
	
	nThreads = min(num_theads, len(attractionPoints))
	workers = [None] * nThreads
	for i in range(nThreads):
		workers[i] = Process(target = calculateAttractionForces, args = (i, q, qResults))
		#worker.setDaemon(True)
		workers[i].start()
	
	while q.qsize() > 0:
		print(":", end="")
		allForces += qResults.get()
	
	while not qResults.empty():
		print("_", end="")
		allForces += qResults.get()
		
	#ipdb.set_trace()
	points += allForces
	
	#print("Min:", min(fs), "Max:", max(fs))
	#print("Mean:", np.mean(fs), "Median:", np.median(fs))
	
	cond = True
	limit = np.mean(np.array(radius) * 2)
	dim = math.sqrt(len(points))
	ps = {}
	while cond and spread > 0:
		print("Spread: %s" % spread)
		cond = False
		for i in range(len(points)):
			p1 = points[i]
# 			if i % 1000 == 0:
# 				print("i: %d" % i)
			pInds = [g for d in range(i - int(dim) * 2, i + int(dim) * 2 + 1, int(dim)) for g in range(d - 2, d + 3)]#[i - 1, i + 1, i - dim, i - dim - 1, i - dim + 1, i + dim, i + dim - 1, i + dim + 1]
			pInds = list(filter(lambda pInd : pInd >= 0 and pInd != i and pInd < len(points), pInds))
			for n in pInds:
				p2 = points[int(n)]
				dist = pointsDistance(p1, p2)
				if dist < limit:
					points[i] -= p2 - p1
					cond = True
					ps[i] = 0
					break
		spread -= 1
	print(ps)
	if csvExport:
		f = open(fileName + ".csv", "w")
	
	for i in range(len(points)):
		if fromCorner:
			distFromCorner = pointsDistance(points[i], [0, 0])
		else:
			distFromCorner = pointsDistance(points[i], center)
			
		if distFromCorner < centerOffsetRm or distFromCorner > max(canvasSize):
			continue
		
		putShape(doc, shapeB, shape, points[i], radius, unit)
		if csvExport:
			f.write(str(points[i][0]) + "\t" + str(points[i][1]) + "\n")
	
	
	if csvExport:
		f.close()
		
	print("File saved to " + fileName)
	doc.save(fileName)
	print('\007')
	sys.exit(0)

def radialPerPerp(fileName, canvasSize, radius, margins, unit, shapeB, poreSizeRange, nStart = 4, \
					   isLongVar = False, isStep = False, centerOffset = 500, longPeriod = 200, fullCircle = 0,
					   scaleDown = 2):
	fileName = fileName.format("{0[0]}-{0[1]}".format(poreSizeRange), nStart, "latvar" if isLongVar else "longlatvar", \
							"step" if isStep else "trig", centerOffset, longPeriod, scaleDown)
	doc = newDoc([0, 0], canvasSize, unit)
	
	minP = min(poreSizeRange); maxP = max(poreSizeRange); meanP = sum(poreSizeRange)/len(poreSizeRange)
	spanP = maxP - minP
	
	r = radius[0] + meanP
	sectAngleSpan = 2 * np.pi/nStart
	defSectAngleStart = 0 #sectAngleSpan / 2
	#maxAlpha = np.pi * 2 - sectAngleSpan
	
	pillars = []
	longSectionStart = 0	
	longSectionEnd = longSectionStart + longPeriod
	nextPoreSizeFactor = 0
	counterCheck = 0
	try:
		while longSectionStart < max(canvasSize):
			sectAngleStart = defSectAngleStart
			
			print("LongSection {}-{}".format(longSectionStart, longSectionEnd))
			
			if isLongVar:
				longLimit = longSectionEnd + longPeriod * meanP/minP
				r = longSectionStart + meanP
			else:
				longLimit = longSectionEnd
			
			if (r + longSectionStart) < maxP + margins[0]:
				r = maxP + margins[0]
				
			while r < longLimit:
				#print("\t\tr {}".format(r))
				color = '#' + str(hex(random.randint(0, 16777215)))[2:]
				alpha = 0
				nextPoreSize = meanP
				while alpha < np.pi * 2: #sectAngleEnd:
					pillars.append([r, alpha, nextPoreSize, longSectionStart, longSectionEnd, color])
					counterCheck += 1
					nextPoreSizeFactor = math.sin(alpha % sectAngleSpan)
					nextPoreSize = (math.sin(alpha * nStart) * spanP/2) + meanP
					#nextPoreSize = nextPoreSizeFactor# * (nextPoreSizeFactor / abs(nextPoreSizeFactor))
					dAlpha = 2 * math.asin((nextPoreSize + margins[0]  * 2) / (2*r))#dAlpha depends on r poresize at alpha
					alpha += dAlpha
			
				
				r += meanP + radius[1]*2
				
# 				sectAngleStart += sectAngleSpan + dAlpha
# 				sectAngleEnd += sectAngleSpan + dAlpha
# 			
			
			longSectionStart += longPeriod
			longSectionEnd = longSectionStart + longPeriod
			nStart *= scaleDown
		raise Exception("Stop loop")
	except Exception as e:
		poreSizes = [p[2] for p in pillars]
		print(e, "alpha:", alpha, "r:", r, "margins:", margins, "nextporesize:", nextPoreSize, "asin:", \
		(nextPoreSize + margins[0]  * 2) / (2*r), min(poreSizes), max(poreSizes), sum(poreSizes) / len(poreSizes))
		pass
	stretchs = []
	if isLongVar:
		for i in range(len(pillars)):
			# lateral center to center distance divided by mean center to center distance
			stretch = (pillars[i][2] + margins[1] * 2) / (meanP + margins[1] * 2)
			# dist to long section start + (dist to long section start * stretch)
			stretchs.append(stretch)
			pillars[i][0] = pillars[i][3] + (pillars[i][0] - pillars[i][3]) * stretch
			#ratios.append(pillars[i][2] / meanP)
			#print(pillars[i][0], r, nextPoreSize / meanP)
			
	pillars[:] = [p for p in pillars if p[0] >= p[3] and p[0] < p[4]]
	
	if csvExport:
		f = open(fileName + ".csv", "w")
	
	for i in range(len(pillars)):
		pointLoc = pol2cart(pillars[i])
		if (not fullCircle and 
		(pointLoc[0] < 0 or pointLoc[0] > canvasSize[0] or pointLoc[1] < 0 or pointLoc[1] > canvasSize[1])) or \
			pillars[i][0] > max(canvasSize) or pillars[i][0] < centerOffset:
			continue
		putShape(doc, shapeB, shape, pointLoc, radius, unit)#, fill = pillars[i][5])
		if csvExport:
			f.write(str(pointLoc[0]) + "\t" + str(pointLoc[1]) + "\n")
			
	if csvExport:
		f.close()
		
	doc.save(fileName)
	print("File saved to " + fileName)
	
def radialPeriodic(fileName, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, isLatDistConst = True, 
		   isStep = False, emptyMiddle = -1, phaseShift = 0):
	global csvExport
	
	"""
	Creates radial regular pillar space with periodic narrowing pores either stepwise or sinusoidal.
	
	Parameters
	----------
	doc : SVG object
		SVG document
	fileName : str
		Path where to save SVG file
	width : float
		document width
	height : float
		document height
	radius : float
		Radius of all pillars
	margins[0] : float
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	margins[1] : float
		Similar to margins[0] but on y axis.
	unit : str
		Spatial unit of the document
	shapeB : ShapeBuilder
		ShapeBuilder object to generate circles
	maxDist : float
		Max empty space between 2 pillars.
	minDist : float
		Min empty space between 2 pillars.
	freq : float
		Space between narrowest pores
	isLatDistConst : bool
		Should the lateral distance stay constant or also vary laterally. Default: True (constant)
	isStep : bool
		Are the varying pore sizes in sinusoidal or step-wise. Default: False (sinusoidal)
	emptyMiddle : float
		Empty space to be left in the middle. Default: -1 meaning no empty space to be left.
	phaseShift : float
		Shift in lateral direction or phase shift. Default: 0 no phase shift.
	"""
	
	fileName = fileName.format(freq)
	
	doc = newDoc([0, 0], canvasSize, unit)
	limit = math.sqrt(canvasSize[0] ** 2 + canvasSize[1] ** 2)/2 # max distance from center (diagonal of the document)
	start = [canvasSize[0] / 2, canvasSize[1] / 2] # starting from the middle
	prevPoint = [0, 0] # previous point
	
	# Adding first point in the middle
	point = start
	
	diameter = radius[0] * 2	
	# Sinusoidal calculation parameters for priodic change of pore sizes
	latLoc = 0
	
	if csvExport:
		f = open(fileName + ".csv", "w")
	
	if latLoc > emptyMiddle:
		putShape(doc, shapeB, shape, point, radius, unit)
		if csvExport:
			f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
		#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, stroke='black', fill='white'))
		
	latDist = diameter + margins[0] * 2
	stepCycle = round(freq / latDist) # At which cycles narrow pores appear
	cycle = 1 # keeping track of cycle
	prevProgress = 0
	while latLoc <= limit:
		progress = round(latLoc/limit * 100)
		if progress > prevProgress + 2:
			print("{}%".format(progress))
			prevProgress = progress
			
		latLoc += latDist
		if isStep:
			if (cycle + round((freq / phaseShift))) % stepCycle == 0:
				longDist = poreSizeRange[0] + diameter 		
			else:	
				longDist = poreSizeRange[1] + diameter
		else:	
			longDist = sinusoidal(poreSizeRange, freq, latLoc + phaseShift) + diameter
		
		if not isLatDistConst:
			latDist = longDist
			
		alpha = 2 * math.asin(longDist / (2 * latLoc))
		
		curAngle = 2 * math.pi
		while curAngle > 0:
			pointLoc = [latLoc * math.cos(curAngle) + start[0], latLoc * math.sin(curAngle) + start[1]]
			if latLoc > emptyMiddle:
				putShape(doc, shapeB, shape, pointLoc, radius, unit)
				if csvExport:
					f.write(str(pointLoc[0]) + "\t" + str(pointLoc[1]) + "\n")
				#doc.addElement(shapeB.createCircle(str(pointLoc[0]) + unit, str(pointLoc[1]) + unit, str(radius) + unit, 0, stroke='black', fill='white'))
			prevPoint = pointLoc
			curAngle -= alpha
		cycle += 1
		
	if csvExport:
		f.close()
		
	doc.save(fileName)
	print("File saved to " + fileName)
	
def axialPeriodic(fileName, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, isLatDistConst = True, isStep = False):
	global csvExport
	"""
	Creates longitudinal regular pillar space with periodic narrowing pores either stepwise or sinusoidal.
	
	Parameters
	----------
	doc : SVG object
		SVG document
	fileName : str
		Path where to save SVG file
	canvasSize: [float, float]
		width and height
	radius : float
		Radius of all pillars
	margins[0] : float
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	margins[1] : float
		Similar to margins[0] but on y axis.
	unit : str
		Spatial unit of the document
	shapeB : ShapeBuilder
		ShapeBuilder object to generate circles
	maxDist : float
		Max empty space between 2 pillars.
	minDist : float
		Min empty space between 2 pillars.
	freq : float
		Space between narrowest pores
	isLatDistConst : bool
		Should the lateral distance stay constant or also vary laterally. Default: True (constant)
	isStep : bool
		Are the varying pore sizes in sinusoidal or step-wise. Default: False (sinusoidal)
	"""
	fileName = fileName.format(freq)
	
	doc = newDoc([0, 0], canvasSize, unit)
	start = [0, 0] # starting from the middle
	
	# Adding first point in the middle
	point = start
	
	diameter = radius[0] * 2	
	# Sinusoidal calculation parameters for priodic change of pore sizes
	latDist = diameter + margins[0] * 2
	stepCycle = round(freq / latDist) # At which cycles narrow pores appear
	cycle = 1 # keeping track of cycle
	if csvExport:
		f = open(fileName + ".csv", "w")
	
	prevProgress = 0
	while point[0] <= canvasSize[0]:	
		progress = round(point[0]/canvasSize[0] * 100)
		if progress > prevProgress + 1:
			print("{}%".format(progress))
			prevProgress = progress
			
		if isStep:
			if cycle % stepCycle == 0:
				longDist = poreSizeRange[0] + diameter 		
			else:	
				longDist = poreSizeRange[1] + diameter
		else:	
			longDist = sinusoidal(poreSizeRange, freq, point[0]) + diameter
			
		if not isLatDistConst:
			latDist = longDist
		
		while point[1] <= canvasSize[1]:
			putShape(doc, shapeB, shape, point, radius, unit)
			if csvExport:
				f.write(str(point[0]) + "\t" + str(point[1]) + "\n")
			#doc.addElement(shapeB.createCircle(str(point[0]) + unit, str(point[1]) + unit, str(radius) + unit, 0, stroke='black', fill='white'))
			point[1] += longDist
			
		cycle += 1
		
		point[1] = 0
		point[0] += latDist
	
	if csvExport:
		f.close()
	doc.save(fileName)
	print("File saved to " + fileName)

def regularMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFileRegular, aroundSpaceFill, alternativeSpace, aroundLimit, aroundSpace
	print('Starting regular mode...')
	choices = interactiveCommands(choiceNames = ["alternative", "around", "around-fill", "around-non-fill", "fill", "normal"], message = "What type (overall pattern area shape) should to be created?", allowAll = True, defaultArgs = defaultArgs)
	
	if choices == -1:
		return
	else:
		if choices[0]:
			regular(svgFileRegular, canvasSize, radius, margins, unit, shapeB, circSpace = alternativeSpace, \
		   fillType = "alternative")
		if choices[1]:
			regular(svgFileRegular, canvasSize, radius, margins, unit, shapeB, circSpace = aroundSpaceFill, \
		   circOuterLimit = aroundLimit, fillType = "around")
		if choices[2]:
			regular(svgFileRegular, canvasSize, radius, margins, unit, shapeB, circSpace = aroundSpace, \
		   circOuterLimit = aroundSpaceFill, fillType ="aroundfill")
		if choices[3]:
			regular(svgFileRegular, canvasSize, radius, margins, unit, shapeB, circSpace = alternativeSpace, \
		   circOuterLimit = aroundLimit, fillType = "aroundnonfill")
		if choices[4]:
			regular(svgFileRegular, canvasSize, radius, margins, unit, shapeB, circSpace = alternativeSpace, \
		   regFill = True, fillType = "fill")
		if choices[5]:
			regular(svgFileRegular, canvasSize, radius, margins, unit, shapeB, fillType = "normal")
	
	print('Regular finished.')

def disarrangedMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFileDisarranged, aroundSpaceFill, alternativeSpace, aroundLimit
	
	print('Starting disarranged...')
	choices = interactiveCommands(choiceNames = ["alternative", "around", "normal"], message = "What type (overall pattern area shape) should to be created?", allowAll = True, defaultArgs = defaultArgs)
	methodChoice = interactiveCommands(choiceNames = disMethods, message = "What randomness method should be created?", allowAll = False, defaultArgs = defaultArgs)
	methodChoice = list(itertools.compress(disMethods, methodChoice))[0]
	scale = float(input("Spread/scale? Enter spread in mm or {0} is default (quarter of pore size in x = {0}): ".format(margins[0]/2)) or margins[0]/2)
	
	if choices == -1:
		return
	else:
		if choices[0]:
			disarranged(svgFileDisarranged, canvasSize, radius, margins, unit, shapeB, scale = scale, method = methodChoice, circSpace = alternativeSpace, fillType = "alternative")
		if choices[1]:
			disarranged(svgFileDisarranged, canvasSize, radius, margins, unit, shapeB, scale = scale, method = methodChoice, circSpace = aroundSpaceFill, circOuterLimit = aroundLimit, fillType = "around")
		if choices[2]:
			disarranged(svgFileDisarranged, canvasSize, radius, margins, unit, shapeB, scale = scale, method = methodChoice, fillType = "normal")
	print('Disarranged finished.')

def polkaMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFilePolka, aroundSpaceFill, alternativeSpace, aroundLimit, aroundSpace
	print('Starting polka dots...')
	choices = interactiveCommands(choiceNames = ["alternative", "around", "normal", "shifted squares"], message = "What type (overall pattern area shape) should to be created?", allowAll = True, defaultArgs = defaultArgs)
	
	if choices == -1:
		return
	else:
		if choices[0]:
			polkaDots(svgFilePolka, canvasSize, radius, margins, unit, shapeB, circSpace = alternativeSpace, fillType = "alternative")
		if choices[1]:
			polkaDots(svgFilePolka, canvasSize, radius, margins, unit, shapeB, circSpace = aroundSpaceFill, circOuterLimit = aroundLimit, fillType = "around")
		if choices[2]:
			polkaDots(svgFilePolka, canvasSize, radius, margins, unit, shapeB, fillType = "normal")
		if choices[3]:
			shiftedPillars(svgFilePolka, canvasSize, radius, margins, unit, shapeB, fillType = "shiftedpillars")
	print('Polka dots finished.')

def shiftedPairsMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFileShiftedPairs, aroundSpaceFill, alternativeSpace, aroundLimit, aroundSpace
	print('Starting shifted pairs...')
	choices = interactiveCommands(choiceNames = ["alternative", "around", "normal"], message = "What type (overall pattern area shape) should to be created?", allowAll = True, defaultArgs = defaultArgs)
	
	if choices == -1:
		return
	else:
		if choices[0]:
			shiftedPairs(svgFileShiftedPairs, canvasSize, radius, margins, unit, shapeB, \
				circSpace = alternativeSpace, fillTye = "alternative")
		if choices[1]:
			shiftedPairs(svgFileShiftedPairs, canvasSize, radius, margins, unit, shapeB, \
				circSpace = aroundSpaceFill, circOuterLimit = aroundLimit, fillType = "around")
		if choices[2]:
			shiftedPairs(svgFileShiftedPairs, canvasSize, radius, margins, unit, shapeB, fillType = "normal")
	print('Polka shifted pairs finished.')

def radialMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFileRadSinusoidalLongVar, svgFileRadSinusoidalLongLatVar, svgFileRadStepLongVar, \
	svgFileRadStepLongLatVar, poreSizeRange, freq
	
	print('Starting radial mode...')
	varChoices = interactiveCommands(choiceNames = ["Longitudinal variability only", \
												 "Lateral and longitudinal variability"], \
				message = "Should the pores be variable in both directions or only one direction?", \
				allowAll = True, defaultArgs = defaultArgs)
	if not isinstance(varChoices, list):
		return
	patternChoices = interactiveCommands(choiceNames = ["Sinusoidal pattern", "Stepwise pattern"], \
									  message = "Which pattern(s) need to be generated?", allowAll = True, \
									  defaultArgs = defaultArgs)
	if not isinstance(varChoices, list):
		return
	
	emptyMiddle = float(input("Empty middle? Enter radius to be empty or -1 is default (no empty middle): ") or "-1")
	phaseShift = float(input("Phase shift? Enter shift distance in mm or 0 is default (no shifting): ") or "-1")
	poreSizeRangeInp = (input("Pore size range from minimum to max in sinus. For stepwise, only min will be considered. Use format e.g. 1.5-2.5 (default {}-{}): ".format(poreSizeRange[0], poreSizeRange[1]))
					  or "-".join([str(x) for x in poreSizeRange]))
	poreSizeRange = [float(x) for x in poreSizeRangeInp.split("-")]
	
	if varChoices == -1 or patternChoices == -1:
		return
	else:
		if varChoices[0] and patternChoices[0]: # lat distance constant with sinusoidal pattern
			radialPeriodic(svgFileRadSinusoidalLongVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		  isLatDistConst = True, isStep = False, emptyMiddle = emptyMiddle, phaseShift = phaseShift)
		if varChoices[0] and patternChoices[1]: # lat distance constant with stepwise pattern
			radialPeriodic(svgFileRadStepLongVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		  isLatDistConst = True, isStep = True, emptyMiddle = emptyMiddle, phaseShift = phaseShift)
		if varChoices[1] and patternChoices[0]: # lat distance variable with sinusoidal pattern
			radialPeriodic(svgFileRadSinusoidalLongLatVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		  isLatDistConst = False, isStep = False, emptyMiddle = emptyMiddle, phaseShift = phaseShift)
		if varChoices[1] and patternChoices[1]: # lat distance variable with stepwise pattern
			radialPeriodic(svgFileRadStepLongLatVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		  isLatDistConst = False, isStep = True, emptyMiddle = emptyMiddle, phaseShift = phaseShift)
		
	print('Radial sinusoidal finished.')
	
def axialMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFileAxialSinusoidalLongVar, svgFileAxialSinusoidalLongLatVar, svgFileAxialStepLongVar, \
	svgFileAxialStepLongLatVar, poreSizeRange, freq
	print('Starting axial mode...')
	
	varChoices = interactiveCommands(choiceNames = ["Longitudinal variability only", \
												 "Lateral and longitudinal variability"], \
				message = "Should the pores be variable in both directions or only one direction?", allowAll = True, \
				defaultArgs = defaultArgs)
	if not isinstance(varChoices, list) and varChoices < 0:
		return
	patternChoices = interactiveCommands(choiceNames = ["Sinusoidal pattern", "Stepwise pattern"], \
									  message = "Which pattern(s) need to be generated?", allowAll = True, \
									  defaultArgs = defaultArgs)
	if not isinstance(patternChoices, list) and patternChoices < 0:
		return
	poreSizeRangeInp = (input("Pore size range from minimum to max in sinus. For stepwise, only min will be considered. Use format e.g. 1.5-2.5 (default {}-{}): ".format(poreSizeRange[0], poreSizeRange[1])) or "-".join([str(x) for x in poreSizeRange]))
	poreSizeRange = [float(x) for x in poreSizeRangeInp.split("-")]
	
	freq = float(input("Frequency of stepwise or sinusoidal pore size change over distance. " \
					"Default: {}: ".format(freq)) or freq)
	
	if varChoices == -1 or patternChoices == -1:
		return
	else:
		if varChoices[0] and patternChoices[0]: # lat distance constant with sinusoidal pattern
			axialPeriodic(svgFileAxialSinusoidalLongVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		 isLatDistConst = True, isStep = False)
		if varChoices[0] and patternChoices[1]: # lat distance constant with stepwise pattern
			axialPeriodic(svgFileAxialStepLongVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		 isLatDistConst = True, isStep = True)
		if varChoices[1] and patternChoices[0]: # lat distance variable with sinusoidal pattern
			axialPeriodic(svgFileAxialSinusoidalLongLatVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		 isLatDistConst = False, isStep = False)
		if varChoices[1] and patternChoices[1]: # lat distance variable with stepwise pattern
			axialPeriodic(svgFileAxialStepLongLatVar, canvasSize, radius, margins, unit, shapeB, poreSizeRange, freq, \
		 isLatDistConst = False, isStep = True)
		
	print('Axial mode finished.')


def radialPerPerpMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	print('Starting radial periodic perpendicular mode...')
	inputs = [["poreSizeRange", str, "Pore size range from minimum to max in sinus. For stepwise, only min will be considered. Use format e.g. 1.5-2.5 (default {}-{}): ".format(poreSizeRange[0], poreSizeRange[1]), 
			"-".join([str(x) for x in poreSizeRange])], 
		   ["nStart", int, "How many angular sections to start with? e.g. 4 (default): ", "4"],
		   ["isLongVar", bool, "Should also longitudinal pore size vary? 0 (default) or 1 : ", 0], 
		   ["isStep", bool, "Step-wise instead of sinusoidal? 0 or 1 (default): ", 0], 
		   ["centerOffset", float, "Center offset (remove pillar before the offset) default 500: ", 500], 
		   ["longPeriod", float, "Longitudinal period (divide sections longitudinally and double number of angular sections at each longitudinal section): default 200", 200],
		   ["scaleDown", float, "Scale down? How much at each row branching need to be scaled? 0-1: branching "\
	"becomes larger and larger, 1: branch size stays the same, >1 branches will get smaller? or 2 is default (2x "\
	"scaling): ", "2"],
		   ["fullCircle", float, "Full circle is 4 quarters, where each quarter is in canvas size. If 0 (default), only a quarter of circle is generated.", 0]]
	
	vals, defaultArgs = getArguments(inputs, defaultArgs)
	#print('7|' + '|'.join([str(float(val)) for val in vals.values()]))
	print('7|' + valsFormatted(list(vals.values())))
	
	radialPerPerp(svgFileRadPeriodicPerp, canvasSize, radius, margins, unit, shapeB, \
					[float(r) for r in vals['poreSizeRange'].split("-")], nStart = vals['nStart'], isStep = vals['isStep'], \
					isLongVar = vals['isLongVar'], centerOffset = vals['centerOffset'], \
					longPeriod = vals['longPeriod'], fullCircle = vals['fullCircle'], scaleDown = vals['scaleDown'])
	print('Fadial periodic perpendicular mode finished.')
	
def branchingMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = []):
	global svgFileRadBranching, svgFileAxBranching, poreSizeRange, freq
	print('Starting radial mode...')
	inputs = [["radialOrAxial", int, "Radial[0] or axial[1] design? (1 axial): ", "1"], 
		   ["forceFunc", int, "Which force function? Gaussian [g]/[0] (default), trigonometric [t]/[1] or mexicanhat"\
	  "[m],[2]?: ", "0"], 
			["centerOffset", float, "Center offset? How much further from center should branching start? or 200 is "\
	"default (200 center offset): ", "200"],
			["centerOffsetRm", float, "Center offset to remove center pillars? How much further from center should pillars to be removed? or 200 is "\
	"default (200 center offset): ", "200"],
			["startNumber", float, "Start number? How many attraction points (pillars close to each other) need to "\
	"be? or 6 is default (6): ", "6"],
			["rowRepeat", float, "Rows can be repeated without reducing the size. How many times do they need to be "\
	"repeated? or 1 (no repeat) is default (1): ", "1"],
			["shiftRows", float, "Rows can be shifted to prevent alignment. 1 (shift) or 0 (aligned) is default "\
	"(1): ", "1"],
			["scaleDown", float, "Scale down? How much at each row branching need to be scaled? 0-1: branching "\
	"becomes larger and larger, 1: branch size stays the same, >1 branches will get smaller? or 2 is default (2x "\
	"scaling): ", "2"],
			["strengthFactor", float, "Strength factor? It is a multiplier of attraction forces. Default is 30: ", \
	"30"],
			["fromCorner", int, "Should the attraction points be laid from the corner? (1 or 0) Default is 1: ", \
	"1"],
			["spread", int, "Should the dots be spread out if they are overlapping? (default is 0)", "0"],
			["closestPillar", bool, "Should attraction points be the closest pillar only? (default is 0)", 0]]#,

	vals, defaultArgs = getArguments(inputs, defaultArgs)
		
	forceFuncTitle = ""
	if vals['forceFunc'] == 0 or vals['forceFunc'] == "gaussian" or vals['forceFunc'] == "g":
		forceFuncTitle = "gaussian"
		vals['forceFunc'] = 0
	elif vals['forceFunc'] == 1 or vals['forceFunc'] == "trigonometric" or vals['forceFunc'] == "t":
		forceFuncTitle = "trigonometric"
		vals['forceFunc'] = 1
	elif vals['forceFunc'] == 2 or vals['forceFunc'] == "mexicanhat" or vals['forceFunc'] == "m":
		forceFuncTitle = "mexicanhat"
		vals['forceFunc'] = 2
		
	#print('6|' + '|'.join([str(float(val)) for val in vals.values()]))
	print('6|' + valsFormatted(list(vals.values())))
	
	
	if(vals['radialOrAxial'] == 0):
		branchingRadial(svgFileRadBranching, canvasSize, radius, margins, unit, shapeB, [i/2 for i in canvasSize], \
				  vals['forceFunc'], vals['rowRepeat'], vals['shiftRows'], vals['centerOffset'], vals['startNumber'],\
					  vals['scaleDown'], strengthFactor = vals['strengthFactor'], forceFuncTitle = forceFuncTitle, \
					  fromCorner = vals['fromCorner'], centerOffsetRm  = vals['centerOffsetRm'], spread = vals['spread'], \
					  closestPillar = vals['closestPillar'])
	else:
		branchingAxial(svgFileAxBranching, canvasSize, radius, margins, unit, shapeB, [i/2 for i in canvasSize], \
				 vals['forceFunc'], vals['rowRepeat'], vals['shiftRows'], vals['centerOffset'], vals['startNumber'], \
				 vals['scaleDown'], strengthFactor = vals['strengthFactor'], forceFuncTitle = forceFuncTitle)	
		
	print('Radial sinusoidal finished.')


def branchingAxial(fileName, canvasSize, radius, margin, unit, shapeB, center, forceFunc = 0, repeatRow = 1, \
				   shiftRows = 1, centerOffset = 0, startNumber = 6, scaleDown = 1.0, k = 1.0, strengthFactor = 30, \
					   forceFuncTitle = "", fromCorner = 1, centerOffsetRm = 0):
	global addIndicators, csvExport
	"""
	Generates a SVG file of regular grid of pillars.
	
	Parameters
	----------
	fileName : str
		Path where to save SVG file
	canvasSize : np array
		document width and height
	radius : float
		Radius of all pillars
	margin : np array
		Margin from outer edge of each pillar towards the outer edge of the pillar space (rectangular space of each pillar) on x axis. This is half the distance between neighbouring pillars in a regular grid.
	unit : str
		Spatial unit of the document
	shapeB : ShapeBuilder
		ShapeBuilder object to generate circles
	center : np array
		center position
	centerOffset : float
		starting from center
	startNumber : int
		number of attraction centers in the first row
	scaleDown : float
		factor by to reduce size of blobs each row
	Returns
	-------
	float[][]
		A two-dimensional array, where first dimension defines each point and second dimension represents axes (x and y)
	"""
	
	fileName = fileName.format(startNumber, scaleDown, centerOffset, centerOffsetRm, forceFuncTitle[:4], \
							strengthFactor, repeatRow, "shift" if bool(shiftRows) else "noshift")
	
	doc = newDoc([0, 0], canvasSize, unit)
	points = np.asarray(regularGrid(radius, margins))
	#print(points.shape)
	attractionPoints = np.empty([0, 3])
	
	rowR = centerOffset
	
	nAPointsInRow = startNumber
	
	rowRepeatCounter = repeatRow
	
	strength = (canvasSize[1] / startNumber)/2
	
	dy = canvasSize[1] / startNumber #dAlpha = 2 * np.pi / startNumber
	
	prevYOffset = strength
	
	while rowR < canvasSize[0] + strength: # and rowR < canvasSize[1]:
		if prevYOffset != 0 or shiftRows == 0:
			y = 0
		else:
			y = strength
			
		prevYOffset = y
		
		while y < (canvasSize[1] + strength):
			#polarCoord = [rowR, alpha]
			
			cartCoord = np.asarray([rowR, y])#pol2cart(polarCoord))
			if not bool(fromCorner):
				cartCoord = cartCoord + center
			cartCoord = np.append(cartCoord, strength)
			if addIndicators:
				putShape(doc, shapeB, shape, cartCoord, strength, unit, fill = "#00FF00", opacity = 0.5)
				putShape(doc, shapeB, shape, cartCoord, radius, unit, fill = "red")
			
			#print(polarCoord, cartCoord)
			#ipdb.set_trace()
			attractionPoints = np.vstack((attractionPoints, cartCoord))
			y += dy #alpha += dAlpha
		
		print(rowR, "/", canvasSize[1], (cartCoord[0]**2 + cartCoord[1]**2)**0.5, strength, rowRepeatCounter, len(attractionPoints))
				
		if rowRepeatCounter == 1:
			rowRepeatCounter = repeatRow
#			nAPointsInRow = round(nAPointsInRow * scaleDown)
			
			rowR += strength + strength / scaleDown
			strength /= scaleDown
		else:
			rowRepeatCounter -= 1
			rowR += strength + strength
		
		nAPointsInRow = (canvasSize[1] / strength) / 2
		
		dy = canvasSize[1] / nAPointsInRow
		#circum = 2 * np.pi * rowR
		
		if strength < 2 * max(radius):
			break
	allForces = np.zeros((len(points), 2))
	fs = []
	#ipdb.set_trace()
	for i in range(len(attractionPoints)):
	#for i in range(50):
		if i % 50 == 0:
			print("F: ", i, "/", len(attractionPoints))
		aPoint = attractionPoints[i]
		strength = aPoint[2]
		forces = np.zeros((len(points), 2))
		#ipdb.set_trace()
		for n in range(len(points)):
			#ipdb.set_trace()
			point = points[n]
			r = pointsDistance(aPoint, point)
			f = getAttractionForce(forceFunc, strength, r)
			
			fs.append(f)				
			vec = aPoint[0:2] - point
			normVec = vec / norm(vec)
			forces[n] = minVec(strengthFactor * strength * f * normVec, vec)
			
		allForces += forces
	#ipdb.set_trace()
	points += allForces
	
	print("Min:", min(fs), "Max:", max(fs))
	print("Mean:", np.mean(fs), "Median:", np.median(fs))
	if csvExport:
		f = open(fileName + ".csv", "w")
		
	for i in range(len(points)):		
		putShape(doc, shapeB, shape, points[i], radius, unit)
		if csvExport:
			f.write(str(points[i][0]) + "\t" + str(points[i][1]) + "\n")
	if csvExport:
		f.close()
		
	print("File saved to " + fileName)
	doc.save(fileName)
	print('\007')
	sys.exit(0)

if __name__ == '__main__':
	parseArgs(sys.argv)
	prepareFileNames()
	while True:
		choices = interactiveCommands(choiceNames = ["reg", "dis", "pol", "shp", "radial", "axial", "branch", \
											   "radPerPerp"], message = "What needs to be created?", \
								allowAll = True, defaultArgs = defaultArgs)
		
		if choices == -1:
			sys.exit()
		else:
			if choices[0]:
				regularMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[1]:
				disarrangedMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[2]:
				polkaMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[3]:
				shiftedPairsMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[4]:
				radialMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[5]:
				axialMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[6]:
				branchingMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)
			if choices[7]:
				radialPerPerpMode(canvasSize, radius, margins, unit, shapeB, manual = True, defaultArgs = defaultArgs)