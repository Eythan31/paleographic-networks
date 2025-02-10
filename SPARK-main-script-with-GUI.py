import os, sys
import csv
import networkx as nx
import math
import matplotlib.pyplot as plt
import datetime
import shutil
import PIL
import copy
import random
import scipy
import traceback
from matplotlib.lines import Line2D
import numpy as np
#from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QDialog, QTableWidgetItem, QFileDialog
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi

###############################################################################
########################## PARAMETERS #########################################
###############################################################################

################################
#CHOICE OF LETTERS AND FEATURES
################################
LETTERS_TO_KEEP = []
#LETTERS_TO_KEEP = []
LETTERS_TO_KEEP.append("א")
LETTERS_TO_KEEP.append("ב")
#LETTERS_TO_KEEP.append("ג")
#LETTERS_TO_KEEP.append("ד")
#LETTERS_TO_KEEP.append("ה")
#LETTERS_TO_KEEP.append("ו")
#LETTERS_TO_KEEP.append("ז")
#LETTERS_TO_KEEP.append("ח")
#LETTERS_TO_KEEP.append("ט")
#LETTERS_TO_KEEP.append("י")
#LETTERS_TO_KEEP.append("כ")
LETTERS_TO_KEEP.append("ל")
LETTERS_TO_KEEP.append("מ")
#LETTERS_TO_KEEP.append("נ")
#LETTERS_TO_KEEP.append("ס")
#LETTERS_TO_KEEP.append("ע")
#LETTERS_TO_KEEP.append("פ")
#LETTERS_TO_KEEP.append("צ")
#LETTERS_TO_KEEP.append("ק")
#LETTERS_TO_KEEP.append("ר")
#LETTERS_TO_KEEP.append("ש")
#LETTERS_TO_KEEP.append("ת")

BODY_PARTS_TO_KEEP = []
#BODY_PARTS_TO_KEEP.append("body:inclination")
BODY_PARTS_TO_KEEP.append("head")
BODY_PARTS_TO_KEEP.append("head:shape")
#BODY_PARTS_TO_KEEP.append("head:angle")#for gimmel and pe
BODY_PARTS_TO_KEEP.append("foot:shape")#for lamed
#BODY_PARTS_TO_KEEP.append("nose")


EXCLUDED_FEATURES = ["מ:head:shape:curved fork", "א:head:u-shaped"] #features to be excluded from the analysis
SINGLE_LETTER = (len(LETTERS_TO_KEEP)==1) #Set to True if comparing features of a single letter, to False otherwise (important for the layout of the nodes)
SHOW_ANTI_EDGES = False #Show an edge only if the two features NEVER occur together (Default=False)

################################
#INPUT FILES
################################
#SCRIPT_FILE = "Analyze-Graphs-2-different-letters.py"
SCRIPT_FILE = "SPARk-main-script-with-GUI.py"
INPUT_FILENAME = "data/features-by-graphs-sorted-by-graph.csv" if SINGLE_LETTER else "data/features-by-inscriptions-sorted_by_feature_2.csv"
print(INPUT_FILENAME)
CSV_DELIMITER = ","

################################
#OUTPUT PARAMETERS
################################
TITLE = "Paleographic network"
WITH_LEGEND = False #with or without legend at the bottom of the output graphs
OUTPUT_FILENAME_PREFIX = "TEST"#used for title of the plot and for name of directory
#THRESHOLD = 3 #obsolete
THRESHOLDS = [1, 3, 5, 7, 9]#generates networks for all these threshold values (these are thresholds on the weight of the edge). If an edge has weight strictly less than the threshold, the edge is removed from the graph.
NODE_THRESHOLD = 0 #Threshold for nodes: if a node has strictly less attestations than this threshold, it is removed from the graph#NOT YET IMPLEMENTED
#FILENAME = "data/comparison-head-alef-bet-mem.csv"
VERBOSE = False

OUTPUT_GRAPH_FILENAME = "output"
OUTPUT_GRAPH_FILENAME_EXTENSION = "jpg"
DRAW_TABLE = False

ALL_COLORS = ['pink', 'blue', 'red', 'grey', 'green', 'orange',  'brown', 
              'black', 'black', 'black', 'black', 'black','black','black',
              'black', 'black', 'black', 'black', 'black','black','black']
SHOW_ADDITIONAL_FEATURES = False#Obsolete

MAX_NODE_SIZE = 400 #500
MIN_NODE_SIZE = 30
MAX_EDGE_WIDTH = 6
MIN_EDGE_WIDTH = 1
LEFT_LIMIT = -2.8
RIGHT_LIMIT = 2.8#3
UPPER_LIMIT = 3.2#2
LOWER_LIMIT = -3.3
PLOT_X_MARGIN = 0.2
PLOT_PAD_INCHES = 0.3
ALPHA = 1# transparency (1 for no transparency)
LEFTMOST_VERTEX_LABELS_OFFSET = 0.6
RIGHTMOST_VERTEX_LABELS_OFFSET = 0.55
NODE_LABELS_VERTICAL_OFFSET = 0.19 #vertical offset for node labels
MULTIPARTITE_LAYOUT_SCALE = 1
NODES_Y_FACTOR = 2.8#1.5 #multiplicative factor for the y coordinate of all vertices
LAYOUT = "multipartite" #Possible values: "circular", "multipartite", "kamada_kawai", "planar", "random", "spectral", "spring", "shell".

#DRAW_WITH_IMAGES = True # Image for graph nodes
#DRAW_WITH_IMAGES = False
icons = {
    "א:head:shape:parallel": "images/aleph-head-parallel.png",
    "א:head:shape:V-shaped":"images/aleph-head-triangular.png",
    "א:head:shape:U-shaped":"images/aleph-head-u-shaped.png",
    "ב:head:shape:quarter-circle" :  "images/beth-head-quarter-circle.png",
    "ב:head:shape:triangular" :  "images/beth-head-triangular.png",
    "ב:head:shape:round" :  "images/beth-head-round.png",
    "ג:head:angle:right" :  "images/gimmel-head-right.png",
    "ג:head:angle:acute" :  "images/gimmel-head-acute.png",
    "ד:head:shape:quarter-circle" :  "images/dalet-head-quarter-circle.png",
    "ד:head:shape:triangular" :  "images/dalet-head-triangular.png",
    "ד:head:shape:round" :  "images/dalet-head-round.png",
    "ו:head:shape:square cup" :  "images/waw-head-square.png",
    "ו:head:shape:V-shaped" :  "images/waw-head-V-shaped.png",
    "ו:head:shape:U-shaped" :  "images/waw-head-U-shaped.png",
    "ו:head:shape:zigzag-shaped" :  "images/waw-head-zigzag.png",
    "ל:foot:shape:angular" : "images/lamed-foot-angular.png", 
    "ל:foot:shape:curved":"images/lamed-foot-curved.png", 
    "ל:foot:shape:single stroke":"images/lamed-foot-single-stroke.png", 
    "מ:head:shape:crossed parallels": "images/mem-crossed-parallel-head.png", 
    "מ:head:shape:square": "images/mem-square-fork-head.png", 
    "מ:head:shape:zigzag":"images/mem-zigzag-head.png", 
    "מ:head:shape:uncrossed parallels":"images/mem-uncrossed-parallel-head.png"  
}
#icons = {}
########################################################################
########### CHANGE WORKING DIRECTORY TO SCRIPT DIRECTORY ###############
########### Necessary for notepad++ to find the right data files #######
###########(otherwise he searches in the Python executable directory ###
########### instead of the script's directory) #########################
########################################################################
scriptdir = os.path.abspath(os.path.dirname(sys.argv[0]))
os.chdir(scriptdir)
########################################################################

try:
    images = {k: PIL.Image.open(fname) for k, fname in icons.items()}# Load images
    images["other"] = PIL.Image.open("images/circle.png")
except Exception as e:
    print("Exception while loading images: ", e)   
    print(traceback.format_exc())


###############################################################################
########################## CODE ###############################################
###############################################################################

#####################################
######## Helper functions ###########
#####################################
def get_all_hebrew_letters():
    letters = []
    letters.append("א")
    letters.append("ב")
    letters.append("ג")
    letters.append("ד")
    letters.append("ה")
    letters.append("ו")
    letters.append("ז")
    letters.append("ח")
    letters.append("ט")
    letters.append("י")
    letters.append("כ")
    letters.append("ל")
    letters.append("מ")
    letters.append("נ")
    letters.append("ס")
    letters.append("ע")
    letters.append("פ")
    letters.append("צ")
    letters.append("ק")
    letters.append("ר")
    letters.append("ש")
    letters.append("ת")
    return letters

#####################################
######### Print functions ###########
#####################################
#Print nicely a matrix implemented as a two-dimentional Python list.
def print_matrix(M):
    n = len(M)
    for a in range(0,n):
        m = len(M[0])
        for b in range(0,m):
            print(M[a][b],  end="  ")
        print()
                
def print_dictionary(d):
    for keys,values in d.items():
        print(keys, ": ", end="")
        print(values)        

########################################################
######## input & data preparation functions ############
########################################################

#Reads a two-column CSV file of inscriptions or graphs (col. 1) and paleographic features (col. 2)
#exported from the nodegoat database, and puts it in a two-dimensional Python 
#list (matrix). Format example: Col. 1=1011 (inscr. number); Col. 2="א:additional:curved lower crossbar"
def read_csv_to_list(inputFileName):    
    with open(inputFileName, newline='', encoding="utf8") as f:
        reader = csv.reader(f, delimiter=CSV_DELIMITER)
        my_list= list(reader)        
    return my_list

#Converts the 2D matrix read from the CSV file to a dictionary where 
#key=inscription and value=list of features attested on that graph. 
#Example: key = 1007; value = ["א:head:parallel"]
def build_features_per_ID_dictionary(l):
    d={}
    for i in range(1, len(l)):#skip first row
        name = l[i][0]
        feature = l[i][1]
        if feature != "":
            if name not in d:
                d[name] = [feature]
            else:
                d[name].append(feature)
    return d

#Converts the 2D matrix read from the CSV file to a dictionary where 
#the key is the full body part (including the letter), 
#and the value is a list of terminal features.
#Example: key = "א:head"; value = ["parallel"]
def build_features_per_body_part_dictionary(l):
    d={}
    for i in range(1, len(l)):#skip first row
        name = l[i][0]
        full_feature = l[i][1]
        
        if full_feature != "":
            letter = full_feature[0]
            body_part = letter + ":" + get_full_body_part(full_feature)
            terminal_feature = get_terminal_feature(full_feature)
            if body_part not in d:
                d[body_part] = [terminal_feature]
            elif terminal_feature not in d[body_part]:
                d[body_part].append(terminal_feature)
    return d

def build_body_parts_per_letter_dictionary(l):
    d={}
    for i in range(1, len(l)):#skip first row
        name = l[i][0]
        full_feature = l[i][1]
        
        if full_feature != "":
            letter = full_feature[0]
            body_part = get_full_body_part(full_feature)
            terminal_feature = get_terminal_feature(full_feature)
            if letter not in d:
                d[letter] = [body_part]
            elif body_part not in d[letter]:
                d[letter].append(body_part)
    return d

def get_terminal_features(d):
    features=[]
    for features_list in d.values():
        for feature in features_list:
            feature_suffix = feature.split(":")[-1]
            if feature_suffix not in features:
                features.append(feature_suffix)
    return features

def get_body_parts(d):
    features=[]
    for features_list in d.values():
        for feature in features_list:
            feature_suffix = feature.split(":")[1]
            if feature_suffix not in features:
                features.append(feature_suffix)
    return features

########################################################
################## statistical functions ###############
########################################################

#Count the total number of occurrences of each feature (independently of the edges)
#and return it in the form of a dictionary. 
#Example: { א:head:parallel : 88 }
def get_feature_count(d):
    count = {}
    for key, val in d.items():#for each graph
        for feature in val:
            suffix = feature#get_feature_no_letter(feature)#get_feature_suffix(feature)
            if suffix not in count:
                count[suffix] = 1
            else:
                count[suffix] += 1
    return count

def get_max_according_to_counts(full_body_part, list_final_features, counts):
    max_ = -1
    for final_feature in list_final_features:
        full_feature = full_body_part + ":" + final_feature
        if counts[full_feature] > max_:
            max_ = counts[full_feature] 
            max_feature = full_feature
    return max_feature      

#Returns a dictionary that associates to each pair of features
#the number of times it occurs. Used for the graph edges. 
#Ex: { ('ב:head:quarter-circle', 'מ:head:square fork') : 3 }
def get_occurrences(d):
    occurrences = {}
    for key in d.keys():#for each graph
        features_list = d[key]
        if(len(features_list)==1):#if that graph has only one feature
            feature = get_feature_no_letter(features_list[0])
        else:
            for i in range(0, len(features_list)):#for each feature of that graph            
                for j in range(0, i):#for each OTHER feature of that graph
                    #feature1 = get_feature_no_letter(features_list[i])
                    #feature2 = get_feature_no_letter(features_list[j])
                    feature1 = features_list[i]
                    feature2 = features_list[j]
                    feature1, feature2 = min(feature1, feature2), max(feature1, feature2)                
                    if (feature1, feature2) not in occurrences:
                        occurrences[(feature1, feature2)] = 1
                    else:
                        occurrences[(feature1, feature2)] += 1
    return occurrences                   

###############################################
######## feature parsing functions ############
###############################################

#Parses a string of features coming from the nodegoat database, beginning
#with the Hebrew letter, followed by substrings separated by a semicolon. 
#Ex: "א:additional:curved lower crossbar"
#Returns a list of parsed items (beginning with the Hebrew letters)
def parse_feature(full_feature):
    if full_feature == "":
        return []
    else:
        return full_feature.split(":")

#Parses a string of features coming from the nodegoat database, beginning
#with the Hebrew letter, followed by substrings separated by a semicolon.
#Ex: "א:additional:curved lower crossbar"
#Returns the terminal feature (the substring after the last semicolon)
def get_terminal_feature(full_feature):
    if full_feature == "":
        return ""
    else:
        return full_feature.split(":")[-1]

#Parses a string of features coming from the nodegoat database, beginning
#with the Hebrew letter, followed by substrings separated by a semicolon.
#Returns the "body part" of the feature (the substring after the first semicolon)
def get_body_part(full_feature):
    if full_feature == "":
        return ""
    else:
        return full_feature.split(":")[1]
    
#Parses a string of features coming from the nodegoat database, beginning
#with the Hebrew letter, followed by substrings separated by a semicolon.
#Returns the full "body part" of the feature (the substring after the first semicolon), 
#i.e. including any prefix.
def get_full_body_part(full_feature):
    if full_feature == "":
        return ""
    else:
        splits = full_feature.split(":")
        if(len(splits)==3):
            return splits[1]
        else:
            return ':'.join(splits[1:-1])

def get_feature_suffix(feature):
    return feature.split(":")[-1] 

def get_feature_no_letter(feature):
    return feature[2:]            

###############################################
############### layout functions ##############
###############################################
    
#gets a list of features, returns their position if drawn horizontally
#on the lower line of the graphic.
def get_positions_lower_line(features):
    pos = {}
    y = -1
    x_left = -0.5
    x_right = 0.5
    offset = 1/(len(features)-1)
    x = x_left
    for feature in features:
        pos[feature] = (x,y)
        x += offset
    return pos

#gets a list of features, returns their position if drawn horizontally
#on the upper line of the graphic.
def get_positions_upper_line(features):
    pos = {}
    y = 1
    x_left = -0.5
    x_right = 0.5
    offset = 1/(len(features)-1)
    x = x_left
    for feature in features:
        pos[feature] = (x,y)
        x += offset
    return pos

#gets a list of features, returns their position if drawn vertically
#on the left side of the graphic.
def get_positions_left_side(features):
    pos = {}
    x=-1    
    y_up = 0.5
    y_low =- 0.5
    offset = 1/(len(features)-1)
    y = y_up
    for feature in features:
        pos[feature] = (x,y)
        y -= offset
    return pos

#gets a list of features, returns their position if drawn vertically
#on the right side of the graphic.
def get_positions_right_side(features):
    pos = {}
    x=1    
    y_up = 0.5
    y_low =- 0.5
    offset = 1/(len(features)-1)
    y = y_up
    for feature in features:
        pos[feature] = (x,y)
        y -= offset
    return pos

# "circular", "multipartite", "kamada_kawai", "planar", "random", "spectral", "spring", "shell"
def get_layout_positions(G):
    if LAYOUT == "circular":
        pos = nx.circular_layout(G)
        for x in pos:
            pos[x][0] *= 1.5#expand points horizontally a bit. Magic number.
        return pos
    elif LAYOUT == "kamada_kawai":
        return nx.kamada_kawai_layout(G)
    elif LAYOUT == "planar":
        return nx.planar_layout(G)    
    elif LAYOUT == "random":
        return nx.random_layout(G)    
    elif LAYOUT == "spectral":
        return nx.spectral_layout(G)    
    elif LAYOUT == "spring":
        return nx.spring_layout(G)  
    elif LAYOUT == "shell":        
        return nx.shell_layout(G)
    elif LAYOUT == "multipartite":        
        pos = nx.multipartite_layout(G, subset_key="layer", scale=MULTIPARTITE_LAYOUT_SCALE)
        #move leftmost points more to the left and rightmost points more to the right
        points = pos.values()
        x_vals = [x[0] for x in points]
        y_vals = [x[1] for x in points]
        min_x = min(x_vals)
        max_x = max(x_vals)
        
        for x in pos:
            if pos[x][0] == min_x:
                pos[x][0] -= 0.3 
            if pos[x][0] == max_x:
                pos[x][0] += 0.3
        return pos

#positions for a square layout (4 categories). Takes the second dictionary as input
def get_positions(d2):
    pos = {}
    if VERBOSE:
        print("test ", d2["head"])
    pos.update(get_positions_upper_line(d2["head"]))            
    pos.update(get_positions_right_side(d2["body"]))
    pos.update(get_positions_lower_line(d2["additional"]))
    return pos

###############################################
############### Graph building functions ######
###############################################

#Returns the list of all the features, identified by their full name.
#Ex: ['א:head:parallel', 'א:head:v-shaped', 'ב:head:quarter-circle']
def get_all_full_feature_names(d2):
    nodes = []
    i = 0
    for key,value in d2.items():
        for x in value:
            nodes.append(key + ":" + x)
            #print("\tappending ", key + ":" + x)
        i += 1
    return nodes 
    
#Variant of the above, where the groups of nodes are sorted from most common to less common
def get_all_full_feature_names_sorted(d, d2):
    counts = get_feature_count(d)
    nodes = []
    i = 0
    for key,value in d2.items():
        value_copy = value[:]
        while len(value_copy ) !=0:            
            max_feature = get_max_according_to_counts(key, value_copy, counts)
            max_feature_final = get_terminal_feature(max_feature)
            #print("\nMAX ACCORDING TO COUNTS = ", max_feature)
            #print("MAX ACCORDING TO COUNTS (FINAL PART) = ", max_feature_final)
            #print("VALUE COPY = ", value_copy)
            value_copy.remove(max_feature_final)
            nodes.append(max_feature)    
    return nodes

def get_node_colors(d2):
    node_color = []
    i=0
    for key,value in d2.items():
        if True:#key != 'crossbars':
            node_color.extend([ALL_COLORS[i] for j in range(0,len(value))])
            i += 1
            
    return node_color

def get_node_colors_from_nodes(nodes):
    node_colors = []
    i=0    
    layer=0
    if not SINGLE_LETTER:#add as a parameter
        while i<len(nodes):
            letter = nodes[i][0]    
            while i<len(nodes) and nodes[i][0]==letter:
                node_colors.append(ALL_COLORS[layer])
                i += 1
            layer+=1
    else:
        while i<len(nodes):            
            body_part = get_full_body_part(nodes[i])
            while i<len(nodes) and get_full_body_part(nodes[i])==body_part:
                node_colors.append(ALL_COLORS[layer])
                i += 1
            if VERBOSE:
                print("Layer ", layer, " = ", body_part)
            layer+=1
    return node_colors

#Add nodes to the graph, with layers for the multipartite layout.
def add_nodes(G, d2, nodes):
    i = 0
    layer=0
    try:
        if not SINGLE_LETTER:#add as a parameter
            while i<len(nodes):
                letter = nodes[i][0]    
                while i<len(nodes) and nodes[i][0]==letter:
                    if nodes[i] in icons:
                        G.add_node(nodes[i], layer=layer, image=images[nodes[i]])
                        #print("found ", nodes[i])
                    else:
                        G.add_node(nodes[i], layer=layer, image=images["other"])
                        print(nodes[i], "not found ")
                    i += 1
                if VERBOSE:
                    print("Layer ", layer, " = ", letter)
                layer+=1
        else:
            while i<len(nodes):            
                body_part = get_full_body_part(nodes[i])
                while i<len(nodes) and get_full_body_part(nodes[i])==body_part:
                    G.add_node(nodes[i], layer=layer)
                    i += 1
                if VERBOSE:
                    print("Layer ", layer, " = ", body_part)
                layer+=1
        # for key,value in d2.items():                
        #     for x in value:
        #         G.add_node(key+":"+x, layer=i)
        #     i += 1
    except Exception as e:
        print("Exception in add_nodes(): ", e)   

        
######################################
    
def draw_table(nodes, edge_labels):
    #matrix for the table
    n=len(nodes)
    matrix = []
    for i in range(0,n):
        matrix.append([])
        for j in range(0,n):  
            feature1 = get_feature_suffix(nodes[i])
            feature2 = get_feature_suffix(nodes[j])            
            if (feature1, feature2) in edge_labels:
                matrix[i].append(edge_labels[(feature1, feature2)])
            else:
                matrix[i].append(0)
                
    n=len(nodes)
    for i in range(0,n):
        for j in range(0,n):            
            feature1 = get_feature_suffix(nodes[i])
            feature2 = get_feature_suffix(nodes[j])            
            if (feature1, feature2) in edge_labels:
                matrix[i][j] = edge_labels[(feature1, feature2)]
                matrix[j][i] = edge_labels[(feature1, feature2)]
    table = plt.table(cellText=matrix, rowLabels=nodes, colLabels=nodes, loc='bottom')
    table.set_fontsize(54)
    table.scale(1.5, 1.5) 

#To be written    
def get_vertex_label_positions_circular_layout(nodes, vertex_pos, node_sizes):
    pass

#get vertex label positions from vertex positions    
def get_vertex_label_positions(nodes, vertex_pos, node_sizes):
    #pos_labels = vertex_pos
    pos_labels = copy.deepcopy(vertex_pos)#to avoid modifying the original pos values. 
    #adjust heights of labels
    for key,value in pos_labels.items():
        index = nodes.index(key)
        offset = node_sizes[index]
        max_node_size = max(node_sizes)
        factor = 1+node_sizes[index]/max_node_size 
        #print("INDEX FOR ", key, " = ", index, ", OFFSET= ",offset)
        if value[1] >=0 : # if in upper half of screen
            pos_labels[key][1] += NODE_LABELS_VERTICAL_OFFSET *factor
        else:
            pos_labels[key][1] -= NODE_LABELS_VERTICAL_OFFSET *factor
    
    #adjust horizontal offset of labels
    points = vertex_pos.values()
    x_vals = [x[0] for x in points]
    y_vals = [x[1] for x in points]
    min_x = min(x_vals)
    max_x = max(x_vals)
    for key,value in pos_labels.items(): #in pos_labels:
        index = nodes.index(key)
        offset = node_sizes[index]
        max_node_size = max(node_sizes)
        factor = 1+node_sizes[index]/max_node_size 
        if pos_labels[key][0] == min_x:
            if value[1] >=0: 
                pos_labels[key][1] -= NODE_LABELS_VERTICAL_OFFSET *factor
            else:
                pos_labels[key][1] += NODE_LABELS_VERTICAL_OFFSET *factor
            pos_labels[key][0] -= 1.8*LEFTMOST_VERTEX_LABELS_OFFSET
        elif pos_labels[key][0] == max_x:
            if value[1] >=0: 
                pos_labels[key][1] -= NODE_LABELS_VERTICAL_OFFSET *factor
            else:
                pos_labels[key][1] += NODE_LABELS_VERTICAL_OFFSET *factor           
            pos_labels[key][0] += 0.1#0.1+RIGHTMOST_VERTEX_LABELS_OFFSET *len(key)/22 
        else:
            pos_labels[key][0] -= 0.8*LEFTMOST_VERTEX_LABELS_OFFSET
    return pos_labels

def get_edge_widths(G, edge_labels):
    if edge_labels != {}: #if there are edges
        weight_list = nx.get_edge_attributes(G,'weight').values()
        max_weight = max(weight_list)
        widths = [ MIN_EDGE_WIDTH + math.floor(x/max_weight*MAX_EDGE_WIDTH ) for x in weight_list]#magic number
        if VERBOSE:
            print("\nWeights list=", weight_list, "len(Weights list) = ", len(weight_list))
    else:
        widths = []
    return widths

def add_edges_to_graph(G, nodes, occurrences, threshold):
    for i in range(0, len(nodes)):#for each pair of terminal features
         for j in range(0, i):          
             feature1 = nodes[i]#get_feature_suffix(nodes[i])
             feature2 = nodes[j]#get_feature_suffix(nodes[j])                          
             feature1, feature2 = min(feature1, feature2), max(feature1, feature2)
             
             if (feature1,feature2) in occurrences and occurrences[(feature1, feature2)] >= threshold:                 
                 G.add_edge(feature1, feature2)
                 G[feature1][feature2]['weight'] = occurrences[(feature1, feature2)]    

# Adds an edge between two node iff these two features never occur together
def add_anti_edges(G, nodes, occurrences):
    for i in range(0, len(nodes)):#for each pair of terminal features
         for j in range(0, i):          
             feature1 = nodes[i]#get_feature_suffix(nodes[i])
             feature2 = nodes[j]#get_feature_suffix(nodes[j])                          
             feature1, feature2 = min(feature1, feature2), max(feature1, feature2)             
             if (feature1,feature2) not in occurrences :
                 G.add_edge(feature1, feature2)
                 G[feature1][feature2]['weight'] = 1

def get_vertex_labels(nodes, feature_count):
    vertex_labels = {}
    for label in nodes:
        vertex_labels[label] = label + " (" + str(feature_count[label]) + ")"
    return vertex_labels

def save_image_to_file(threshold, dir_name, show_edges):
    filename = dir_name + OUTPUT_GRAPH_FILENAME
    if show_edges:
        filename += "_with_threshold_" + str(threshold)  if threshold>1 else "_no_threshold"
    else:
        filename += "_no_edges_"
    filename += "." + OUTPUT_GRAPH_FILENAME_EXTENSION
    #os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.3)
        
def add_legend():
    size=12
    # legend_elements = [
    # Line2D([0], [0], marker='o', color='w', label='Aleph',markerfacecolor='g', markersize=size),
    # Line2D([0], [0], marker='o', color='w', label='Mem',markerfacecolor='r', markersize=size),        
    # Line2D([0], [0], marker='o', color='w', label='Beth',markerfacecolor='b', markersize=size),        
    # ]
    legend_elements = []
    if len(LETTERS_TO_KEEP) >1 :
        for i in range(0, len(LETTERS_TO_KEEP)):
            letter = LETTERS_TO_KEEP[i]
            color = ALL_COLORS[i]
            legend_elements.append(Line2D([0], [0], marker='o', color="w", label=letter,markerfacecolor=color, markersize=size))    
    else:
        for i in range(0, len(BODY_PARTS_TO_KEEP)):
            part = BODY_PARTS_TO_KEEP[i]
            color = ALL_COLORS[i]
            legend_elements.append(Line2D([0], [0], marker='o', color="w", label=part,markerfacecolor=color, markersize=size))            
    plt.legend(handles=legend_elements, loc='lower right')
    
def multiply_pos_y(pos, factor):
    for key in pos:
        pos[key][1]*=factor

#apply a random vertical offset to each point in pos. The offset is between -max_offset and max_offset. Use 0 for no offset.
def random_vertical_offset(pos, max_offset):
    if max_offset != 0:
        for key in pos:
            pos[key][1] += random.random()*(max_offset+1)-1
        
#apply a random horizontal offset to each point in pos. The offset is between -max_offset and max_offset. Use 0 for no offset.
def random_horizontal_offset(pos, max_offset):
    if max_offset != 0:
        for key in pos:
            pos[key][0] += random.random()*(max_offset+1)-1
            
#Removes a list of body parts from the graphs-to-features dictionary
def remove_from_dict(d, body_parts):      
    for key, value in d.items():
        for feature in value:
            for body_part in body_parts:
                if body_part in feature:
                    d[key].remove(feature)
                    
def remove_body_type_from_list(l, body_type):
    l2 = []
    for i in range(1,len(l)): #skip first item
        feature = l[i][1]
        if(get_full_body_part(feature) != body_type ):            
            l2.append(l[i])
    return l2

def remove_body_types_from_list(l, body_types):
    l2 = []
    for i in range(1,len(l)): #skip first item
        feature = l[i][1]
        if(get_full_body_part(feature) not in body_types ):    
            l2.append(l[i])
    return l2

#Filters the list (that was read from CSV) to keep only specific body parts.
#"l" is a 2D list read from CSV, with first column=inscription ID and second column=full feature
#"body_parts_to_keep" is a list of body parts to be kept
#body parts can be simple ("head") or composite ("head:shape").
def filter_body_parts_from_list(l, body_parts_to_keep):
    l2 = []
    for i in range(1,len(l)): #skip first item
        feature = l[i][1]
        body_part = get_full_body_part(feature)
        if(body_part in body_parts_to_keep ):    
            l2.append(l[i])
    return l2

#Filters the list (that was read from CSV) to keep only specific letters.
#"l" is a 2D list read from CSV, with first column=inscription ID and second column=full feature
#"letters" is a list of Hebrew letters (in Hebrew fonts)
#
def filter_letters_from_list(l, letters):
    l2 = []
    for i in range(1,len(l)): #skip first item
        feature = l[i][1]
        letter = feature[0]
        if(letter in letters):                
            l2.append(l[i])
    return l2

def filter_excluded_features_from_list(l, excluded_features):
    l2 = []
    for i in range(1,len(l)): #skip first item
        feature = l[i][1]
        body_part = get_full_body_part(feature)
        if(feature not in excluded_features ):            
            l2.append(l[i])
    return l2

def filter_included_features_from_list(l, included_features):
    l2 = []
    for i in range(1,len(l)): #skip first item
        feature = l[i][1]
        body_part = get_full_body_part(feature)
        if(feature in included_features ):
            l2.append(l[i])
    return l2

#Sorts the list of nodes according to the order of list of letters of 
#LETTERS_TO_KEEP and of BODY_PARTS_TO_KEEP. This allows users to control
#the order of appearance of the nodes in the network (layers) by choosing the
#order in LETTERS_TO_KEEP and of BODY_PARTS_TO_KEEP. 
#Returns the sortes version of the list of nodes
def sort_nodes(nodes):
    sorted_nodes = []
    #if many letters, sort by letter
    if not SINGLE_LETTER:
        for letter in LETTERS_TO_KEEP:
            for node in nodes:
                if node[0] == letter:
                    sorted_nodes.append(node)
    else: #else sort by body parts
        for body_part_to_keep in BODY_PARTS_TO_KEEP:
            for node in nodes:
                body_part = get_full_body_part(node)
                if get_full_body_part(node) == body_part_to_keep:
                    sorted_nodes.append(node)
    return sorted_nodes
    

# if not SHOW_ADDITIONAL_FEATURES:
#     BODY_PARTS_TO_EXCLUDE.append("additional")
def make_output_directory():
    now = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")##   
    filename = "outputs/"                       
    filename += OUTPUT_FILENAME_PREFIX + "_"+now 
    #filename += "_with_threshold" if THRESHOLD>1 else "_no_threshold"
    filename += "/"
    os.makedirs(os.path.dirname(filename), exist_ok=True)    
    return filename


def build_graph(d, d2, show_edges, threshold):
    nodes = get_all_full_feature_names(d2)
    #print("In build_graph(). Nodes 1 = ", nodes)
    #nodes = sort_nodes(nodes)
    #print("In build_graph(). Nodes 2 = ", nodes)
    occurrences = get_occurrences(d)        
    feature_count = get_feature_count(d)    
    
    G = nx.Graph() 
    add_nodes(G, d2, nodes)           

    if show_edges:
        if not SHOW_ANTI_EDGES:
            add_edges_to_graph(G, nodes, occurrences, threshold)
        else:
            add_anti_edges(G, nodes, occurrences)
    return G

def draw_title(show_edges, threshold):
    if show_edges:
        threshold_string = ("threshold=" + str(threshold)) if threshold>1 else "no threshold"
        plt.title(TITLE + " (" + threshold_string +  ")", fontsize=16)
    else:
        #threshold_string = ("threshold=" + str(threshold)) if threshold>1 else "no threshold"
        plt.title(TITLE + " (" + "nodes only" +  ")", fontsize=16)

def draw_vertex_labels(G, nodes, pos, node_sizes, feature_count):
    pos_labels = get_vertex_label_positions(nodes, pos, node_sizes)
    vertex_labels = get_vertex_labels(nodes, feature_count)
    nx.draw_networkx_labels(G, pos=pos_labels, labels=vertex_labels, font_size=8, horizontalalignment="left")
    
def draw_network(d, d2, threshold, dir_name, vertical_offset=0, horizontal_offset=0, show_edges=True, with_images = False):
    #print("In draw_network()")
    nodes = get_all_full_feature_names(d2)    
    occurrences = get_occurrences(d)    
    feature_count = get_feature_count(d)        
    max_feature_count = max(feature_count.values())
    
    G = build_graph(d, d2, show_edges, threshold)
    if VERBOSE:
        print("\nGraph nodes 1=",G.nodes())
    
    pos = get_layout_positions(G)
    multiply_pos_y(pos, NODES_Y_FACTOR)
    edge_labels = nx.get_edge_attributes(G,'weight')  
    widths = get_edge_widths(G, edge_labels) if not SHOW_ANTI_EDGES else 1        
    node_color = get_node_colors_from_nodes(nodes)     
    edge_color = 'grey' if not SHOW_ANTI_EDGES else 'red'    
    node_sizes = [ MIN_NODE_SIZE+ math.floor(feature_count[feature]/max_feature_count*MAX_NODE_SIZE) for feature in nodes]        
    random_vertical_offset(pos, vertical_offset)
    random_horizontal_offset(pos, horizontal_offset)

    #Initialize figure (new version)  
    fig = plt.figure()#do the "fig = plt.figure()" before launching the GUI, otherwise the output GUI and plots are too small.  No, I need to redo it here each time, otherwise the letter images move when I redo the "show" several times
    ax = plt.gca() 
    ax.set_xlim(LEFT_LIMIT, RIGHT_LIMIT) 
    ax.set_ylim(LOWER_LIMIT,UPPER_LIMIT)#to avoid labels being cut    
    plt.margins(x=PLOT_X_MARGIN)#to avoid labels being cut
        
    if not with_images:
        print("pos=", pos)
        print("edge_color", edge_color)
        nx.draw(G, pos, edge_color=edge_color, width=widths, 
            node_color=node_color, 
            linewidths=1,
            node_size=node_sizes, 
            alpha=ALPHA, 
            with_labels=False        
        ) 
    else:
        #Draw edges (with gap in edges for the images)  
        nx.draw_networkx_edges(  
            G, pos=pos, 
            edge_color=edge_color, 
            ax=ax, 
            width=widths,
            arrows=True, 
            arrowstyle="-", 
            min_source_margin=20, 
            min_target_margin=20,
        )
    
    #Draw edge labels
    if not SHOW_ANTI_EDGES and edge_labels != {}: #if there are edges
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=6)
        
    # pos_labels = get_vertex_label_positions(nodes, pos, node_sizes)
    # vertex_labels = get_vertex_labels(nodes, feature_count)    
    # nx.draw_networkx_labels(G, pos=pos_labels, labels=vertex_labels, font_size=8, horizontalalignment="left")  
    draw_vertex_labels(G, nodes, pos, node_sizes, feature_count)
    
    if VERBOSE:
        print("\nNODES=", nodes, "; length =", len(nodes))
        print("\nNODE COLORS=", node_color, "; length =", len(node_color))    
        print("\nGraph nodes 1=",G.nodes())
        print("\nPositions\n",pos)
        print("\nOccurrences = ", occurrences)
        print("\nFEATURE COUNT: ", feature_count)
        print("\n max feature count: ", max_feature_count)
        print("\nNode sizes: ", node_sizes, "len(Node sizes): ", len(node_sizes))
        print("\nEdge labels=", edge_labels, "\len(edge labels) = ", len(edge_labels))
        print("\nWidths=", widths, "len(Widths)=", len(widths))
        print("\nVertex labels: ", vertex_labels)
        
    draw_title(show_edges, threshold)    
    if DRAW_TABLE:
        draw_table(nodes, edge_labels) 

    if with_images:
        ############TEST for including images ##############                   
        # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
        tr_figure = ax.transData.transform
        # Transform from display to figure coordinates
        tr_axes = fig.transFigure.inverted().transform
    
        # Select the size of the image (relative to the X axis)           
        icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015 #icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
        #print("icon size: ", icon_size)
        icon_center = icon_size / 2.0
    
        i=0
        for n in G.nodes:           
            xf, yf = tr_figure(pos[n])
            xa, ya = tr_axes((xf, yf))         
            #print("xf,yf = ", xf, ", ", yf)
            #print("xa,ya = ", xa, ", ", ya)           
            if "image" in G.nodes[n]:
               # get overlapped axes and plot icon   
               a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
               a.imshow(G.nodes[n]["image"])#display the image
            a.axis("off")
            i=i+1
    
    if WITH_LEGEND:
        plt.legend(title='Legend:')#removing legend for now (has to be updated to get the right letters and colors)
        add_legend()   
    save_image_to_file(threshold, dir_name, show_edges)

    plt.show()
    


def draw_network_with_images(d, d2, threshold, dir_name, show_edges=True):           
    nodes = get_all_full_feature_names(d2)      
    nodes = sort_nodes(nodes)        
    occurrences = get_occurrences(d)        
    feature_count = get_feature_count(d)    
    max_feature_count = max(feature_count.values()) 

    if VERBOSE:
        print("\nNODES=", nodes, "; length =", len(nodes))
        print("\nNODES SORTED=", nodes, "; length =", len(nodes))       
        print("\nOccurrences = ", occurrences)
        print("\nFEATURE COUNT: ", feature_count)
        print("\n max feature count: ", max_feature_count)
        
    G = build_graph(d, d2, show_edges, threshold)
    if VERBOSE:
        print("\nGraph nodes 1=",G.nodes())
        
    pos = get_layout_positions(G)
    multiply_pos_y(pos, NODES_Y_FACTOR)       
    edge_labels = nx.get_edge_attributes(G,'weight')    
    widths = get_edge_widths(G, edge_labels)    
    node_color = get_node_colors_from_nodes(nodes)#get_node_colors(d2)
    edge_color = 'grey' if not SHOW_ANTI_EDGES else 'red'  
    node_sizes = [ MIN_NODE_SIZE+ math.floor(feature_count[feature]/max_feature_count*MAX_NODE_SIZE) for feature in nodes]
          
    if VERBOSE:
        print("\nNode sizes: ", node_sizes, "len(Node sizes): ", len(node_sizes))
        print("\nEdge labels=", edge_labels, "\len(edge labels) = ", len(edge_labels))
        print("\nWidths=", widths, "len(Widths)=", len(widths))
        print("\nNODE COLORS=", node_color, "; length =", len(node_color))
        print("\nNode sizes: ", node_sizes, "len(Node sizes): ", len(node_sizes))
    
    #Initialize figure
    fig = plt.figure()    
    ax = plt.gca() #new    
    ax.set_xlim(LEFT_LIMIT, RIGHT_LIMIT) 
    ax.set_ylim(LOWER_LIMIT,UPPER_LIMIT)#to avoid labels being cut    
    plt.margins(x=PLOT_X_MARGIN)#to avoid labels being cut
    
    #Draw edges (with gap in edges for the images)  
    nx.draw_networkx_edges(  
        G, pos=pos, 
        edge_color=edge_color, 
        ax=ax, 
        width=widths,
        arrows=True, 
        arrowstyle="-", 
        min_source_margin=20, 
        min_target_margin=20,
    )
    
    #Draw edge labels
    if not SHOW_ANTI_EDGES and edge_labels != {}: #if there are edges
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=6)        

    draw_vertex_labels(G, nodes, pos, node_sizes, feature_count)
    draw_title(show_edges, threshold)
        
    if DRAW_TABLE:
        draw_table(nodes, edge_labels) 


    ############TEST for including images ##############
    # Transform from data coordinates (scaled between xlim and ylim) to display coordinates
    tr_figure = ax.transData.transform
    # Transform from display to figure coordinates
    tr_axes = fig.transFigure.inverted().transform

    # Select the size of the image (relative to the X axis)
    #icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.025
    icon_size = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015
    #print("icon size: ", icon_size)
    icon_center = icon_size / 2.0

    i=0
    for n in G.nodes:
       xf, yf = tr_figure(pos[n])
       xa, ya = tr_axes((xf, yf))
       
       # get overlapped axes and plot icon        
       #a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
       if "image" in G.nodes[n]:
           #a = plt.axes([xa - node_sizes[i]/10000, ya - node_sizes[i]/10000, node_sizes[i]/5000, node_sizes[i]/5000])
           a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
           a.imshow(G.nodes[n]["image"])
       a.axis("off")
       i=i+1
    ##################################################
    
    if WITH_LEGEND:
        plt.legend(title='Legend:')#removing legend for now (has to be updated to get the right letters and colors)
        add_legend()   
   
    save_image_to_file(threshold, dir_name, show_edges)
    plt.show()
    

#####################################    
################ GUI ################    
#####################################
class Window1(QMainWindow):
    def __init__(self):
       super(Window1, self).__init__()
       
       #loadUi(resource_path("./SPARK-app.ui"), self)
       if getattr(sys, 'frozen', False):
            RELATIVE_PATH = os.path.dirname(sys.executable)
       else:
            RELATIVE_PATH = os.path.dirname(__file__)
       self._ui_path = RELATIVE_PATH #+ "/ui_path"  # Update this as needed                
       loadUi(os.path.join(self._ui_path, 'SPARK-app.ui'), self)
       #self.layoutLabel.adjustSize()
       self.file_opened  = False
        # Open action
       #self.actionOpen.triggered.connect(self.open_clicked)       
       #self.contrastButton.clicked.connect(self.launch_contrast)
       #self.actionSettings.triggered.connect(self.settings_clicked)
       items = body_parts_dict["א"]
       self.putItemOnTopOfList_2(items, "head")
       self.putItemAtBottomOfList_2(items, "additional")
       self.putItemAtBottomOfList_2(items, "ligatures")
       self.bodyPartsComboBox_2.addItems(items)
       self.featuresComboBox_2.addItems(["ALL"])
       self.featuresComboBox_2.addItems(features_dict["א"+":"+body_parts_dict["א"][0]])
       self.letterComboBox_2.activated.connect(self.comboBoxLetterAction)
       self.bodyPartsComboBox_2.activated.connect(self.comboBoxBodyPartsAction)       
       self.addButton.clicked.connect(self.clickedAddButton)
       self.resetButton.clicked.connect(self.clickedResetButton)
       self.removeButton.clicked.connect(self.clickedRemoveButton)
       self.showButton.clicked.connect(self.clickedShowButton)
       self.layoutComboBox_2.activated.connect(self.comboBoxLayoutAction)
       
    def open_clicked(self):
        self.file_opened = True
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","CSV Files (*.csv)", options=options)
        if fileName:
            print("Opening file" + fileName)
        
        #self.new_list = prepare_data(fileName, "test-output.csv")
        #self.compute_all()

    def comboBoxLetterAction(self):
        selected_letter = ""
        try:
            selected_letter = str(self.letterComboBox_2.currentText())                    
            #print("Selected letter: ", selected_letter)
            self.bodyPartsComboBox_2.clear()
            items = body_parts_dict[selected_letter]            
            #put "head" first in the list
            self.putItemOnTopOfList_2(items, "head")
            self.putItemAtBottomOfList_2(items, "additional")
            self.putItemAtBottomOfList_2(items, "ligatures")
            self.bodyPartsComboBox_2.addItems(items)                        
            self.comboBoxBodyPartsAction()
        except Exception as e:
            print("Exception: ", e)
            print(traceback.format_exc())

    def putItemOnTopOfList(self, list_, itemName):        
        if itemName in list_:
            index = list_.index(itemName)
            headItem = list_[index]
            list_.pop(index)#remove from where it was
            list_.insert(0, headItem)#insert at the beginning

    def putItemOnTopOfList_2(self, list_, itemPrefix):
        to_be_moved = []
        for body_part in list_:
            if body_part.startswith(itemPrefix):
                to_be_moved.append(body_part)
        for part in to_be_moved:
            list_.remove(part)#remove from where it was
            list_.insert(0, part)#insert at the beginning

    def putItemAtBottomOfList(self, list_, itemName):
        if itemName in list_:
            index = list_.index(itemName)
            headItem = list_[index]
            list_.pop(index)#remove from where it was
            list_.append(headItem)#insert at the end

    def putItemAtBottomOfList_2(self, list_, itemSuffix):
        to_be_moved = []
        for body_part in list_:
            if body_part.endswith(itemSuffix):
                to_be_moved.append(body_part)
        for part in to_be_moved:
            list_.remove(part)#remove from where it was
            list_.append(part)#insert at the end

    def comboBoxBodyPartsAction(self):
        selected_letter = str(self.letterComboBox_2.currentText()) 
        selectedBodyPart = str(self.bodyPartsComboBox_2.currentText())
        #print("selected body part: ",  selectedBodyPart)
        #print("Searching for ", selected_letter + ":" + selectedBodyPart)
        features = features_dict[selected_letter + ":" + selectedBodyPart]
        #print("found features: ", features)
        self.featuresComboBox_2.clear()
        self.featuresComboBox_2.addItems(["ALL"])
        self.featuresComboBox_2.addItems(features)

    def comboBoxLayoutAction(self):        
        #LAYOUT = str(self.layoutComboBox_2.currentText())
        #print("Layout: ", LAYOUT)
        #print("in comboBoxLayoutAction")
        pass
        
    def clickedAddButton(self):
        try:
            selected_letter = str(self.letterComboBox_2.currentText())
            selectedBodyPart = str(self.bodyPartsComboBox_2.currentText())
            #print("Searching for ", selected_letter + ":" + selectedBodyPart)
            selectedFeature = str(self.featuresComboBox_2.currentText())
            selection = selected_letter + ":" + selectedBodyPart + ":" + selectedFeature
            #print(selection)
            if selectedFeature != "ALL":            
                if not selection in all_selected_features:
                    all_selected_features.append(selection)
                    self.chosenFeaturesListWidget.addItems([selection])
            else: #selectedFeature == "ALL":  
                prefix = selected_letter + ":" + selectedBodyPart 
                features = features_dict[prefix]
                for feature in features:
                    feature_long = prefix + ":" + feature
                    #print("List before if: 3", all_selected_features)
                    if not feature_long in all_selected_features:                    
                        all_selected_features.append(feature_long)
                        #print("adding ", feature_long)
                        self.chosenFeaturesListWidget.addItems([feature_long])
            #print("Selected features: ", all_selected_features)
        except Exception as e:
            print("Exception: ", e)  
            print(traceback.format_exc())

    def clickedResetButton(self):
        all_selected_features.clear()
        self.chosenFeaturesListWidget.clear()
        #print("Removing features. List = ", all_selected_features)

    def clickedRemoveButton(self):
        #print("clicked remove button")
        chosenItem = self.chosenFeaturesListWidget.currentItem()
        if chosenItem != None:
            #print("chosen item: ", chosenItem.text())
            self.chosenFeaturesListWidget.takeItem( self.chosenFeaturesListWidget.row(chosenItem))
            #remove from list (to do)
            all_selected_features.remove(chosenItem.text())
  
    def clickedShowButton(self):
        try:
            #print("selected features: ", all_selected_features)
            global LAYOUT
            LAYOUT = str(self.layoutComboBox_2.currentText())
            threshold = int(self.thresholdLineEdit.text())
            vertical_offset = float(self.verticalOffsetLineEdit.text())
            horizontal_offset = float(self.horizontalOffsetLineEdit.text())                  
            ll = filter_included_features_from_list(l, all_selected_features)
            d  = build_features_per_ID_dictionary(ll) #dictionary of features per inscription (or per graph). Ex: {1007 :["א:head:parallel"]}
            d2 = build_features_per_body_part_dictionary(ll)
            #print_dictionary(d)
            #print()
            #print_dictionary(d2)
            dir_name = make_output_directory()
            global SHOW_ANTI_EDGES
            SHOW_ANTI_EDGES = self.antiEdgesCheckBox.isChecked()            
            DRAW_WITH_IMAGES = self.useImagesCheckBox.isChecked()
            draw_network(d, d2, threshold, dir_name, vertical_offset, horizontal_offset, show_edges=True, with_images=DRAW_WITH_IMAGES)
            
        except Exception as e:
            print("Exception: ", e)   
            print(traceback.format_exc())            
        
#####################################
########### MAIN function ###########
#####################################
def main():
    l = read_csv_to_list(INPUT_FILENAME)
    #l = remove_body_types_from_list(l, BODY_PARTS_TO_EXCLUDE)
    l = filter_body_parts_from_list(l, BODY_PARTS_TO_KEEP)
    l = filter_letters_from_list(l, LETTERS_TO_KEEP)
    l = filter_excluded_features_from_list(l, EXCLUDED_FEATURES)
    # print("List read from CSV", l)   
    d  = build_features_per_ID_dictionary(l) #dictionary of features per inscription (or per graph). Ex: {1007 :["א:head:parallel"]}
    d2 = build_features_per_body_part_dictionary(l)#Ex: {"א:head" : ["parallel"]}  
    dir_name = make_output_directory()
    
    if not DRAW_WITH_IMAGES:
        draw_network(d, d2, 1, dir_name, show_edges=False)
        for threshold in THRESHOLDS:
            draw_network(d, d2, threshold, dir_name, show_edges=True)
    else:    
        #draw_network(d, d2, 7, dir_name, show_edges=True)#for testing purposes only. To be eventually removed from here.
        draw_network_with_images(d, d2, 1, dir_name, show_edges=True)
        #draw_network_with_images(d, d2, 7, dir_name, show_edges=True)
        # draw_network(d, d2, 1) # drawing without threshold
        # draw_network(d, d2, THRESHOLD) # drawing with threshold
        #copy source file to output directory
        shutil.copyfile('./'+SCRIPT_FILE, "./"+dir_name+SCRIPT_FILE)
        print("...done.")
        
######################################
#### OLD, non-GUI version#############
######################################
#main()
######################################

######################################
# New GUI version GUI
######################################
l = read_csv_to_list(INPUT_FILENAME)
body_parts_dict = build_body_parts_per_letter_dictionary(l)
features_dict   = build_features_per_body_part_dictionary(l)#Ex: {"א:head" : ["parallel"]}
all_selected_features = []
#print_dictionary(features_dict)
try:
    fig = plt.figure()#do this here, before launching the GUI, otherwise the output GUI and plots are too small.
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = Window1()
    mainWindow.show()
    app.exec_()
    x = input()
except Exception as e:
    print("Exception: ", e)
    print(traceback.format_exc())
