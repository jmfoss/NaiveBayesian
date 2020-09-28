#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:45:37 2019

@author: jmfoss
"""
import json
import pandas as pd

def py_nb():
    while(True):
        choice = input('1) Naive Bayesian classifier from data\n2) Load and test accuracy of Naive Bayesian classifier\n3) Apply a Naive Bayesian classifier to new cases\n4) Quit\nSelect by number: ')
        if choice == '1':
            loadData()
        elif choice == '2':
            test()
        elif choice == '3':
            choice = input('1) Enter a new case interactively\n2) Quit\nSelect by number: ')
            if choice == '1':
                interactive()
        else:
            return;
        
def loadData():
    filename = input('Input file name of data in ARFF format: ')
    file = open(filename, 'rt')
    data = False
    content = file.readlines()
    attributes = []
    attributeNames = []
    for line in content:
        line = line.strip()
        if not data:
            if '@attribute' in line.lower():
                attributeNames.append(line.split()[1])
                attributeSet = line[line.find("{")+1:line.find("}")].replace(' ', '').split(',')
                attributeOptions = {}
                for word in attributeSet:        
                    attributeOptions[word] = 0
                attributes.append(attributeOptions)
            if '@data' in line.lower():
                data = True
                for at in attributes[:-1]:
                    for k, v in at.items():
                        at[k] = attributes[-1].copy()
                    
        else:
            if line and (line[0] != '%'):
                training = line.split(',')
                result = training[-1].strip()
                attributes[-1][result] += 1
                for condition, location in zip(training[:-1], attributes[:-1]):
                    location[condition.strip()][result] += 1
    file.close()
    for value in attributes[:-1]:
        for k, v in value.items():
            for dk, dv in attributes[-1].items():
                value[k][dk] = v[dk] / dv
    total = 0
    for k, v in attributes[-1].items():
        total += v
    for k, v in attributes[-1].items():
        attributes[-1][k] = attributes[-1][k] / total
    saveFile = open('model_' + filename.split('.')[0] + '.bin', 'w')
    attributes.insert(0, attributeNames)
    json.dump(attributes, saveFile)
    print("\nModel saved as: " + 'model_' + filename.split('.')[0] + '.bin' + "\n")
    saveFile.close()
    return;

def loadModel():
    filename = input('Input file name of previously saved model: ')
    file = open(filename, 'rt')
    model = json.load(file)
    attributeNames = model.pop(0)
    return model, attributeNames;
    
def test():
    model = loadModel()[0]       
    filename = input('Input file name of testing data: ')
    file = open(filename, 'rt')
    content = file.readlines()
    data = False
    predictions = model[-1].copy()
    for k, v in predictions.items():
        predictions[k] = model[-1].copy()
        for k1, v1 in predictions[k].items():
            predictions[k][k1] = 0
    for line in content[1:]:
        if not data:
            if '@data' in line.lower():
                data = True
        else:     
            probability = {}
            for k, v in model[-1].items():
                probability[k] = 1
            if line.strip():
                testingValues = line.strip().split(',')
                for condition, value in zip(model[:-1], testingValues[:-1]):
                    for dk, dv in model[-1].items():
                        probability[dk] *= (condition[value][dk])
                for dk, dv in model[-1].items():
                    probability[dk] *= dv
                    highest = max(probability, key=probability.get)
                predictions[highest][testingValues[-1]] += 1
    print_matrix(predictions) 
    
def print_matrix(predictions):
    print("\n Confusion Matrix:\n")
    data = pd.DataFrame(predictions)
    print(data.iloc[::-1])
            
def interactive():
    modelWithNames = loadModel()
    attributeNames = modelWithNames[1]
    model = modelWithNames[0]
    while(True):
        case = ""
        for attribute, name in zip(model[:-1], attributeNames[:-1]):
            print("Select one of the conditions for", name, ":", end = ' ')
            for k, v in attribute.items():
                print(k, end = ' ')
            case += input()
            case+= ','
        probability = {}
        for k, v in model[-1].items():
            probability[k] = 1
        if case.strip():
            testingValues = case.strip().split(',')
            for condition, value in zip(model[:-1], testingValues[:-1]):
                for dk, dv in model[-1].items():
                    probability[dk] *= (condition[value][dk])
            for dk, dv in model[-1].items():
                probability[dk] *= dv
                highest = max(probability, key=probability.get)
            print("\n", "Prediction decision for", attributeNames[-1], ":", highest)
        choice = input("Would you like to input another case y/n: ")
        if choice == 'n':
            return;
    return;
        
        
            
            