import json
import os
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as gemetric_data


def getEdges(records, isGlobal=True) -> tuple:
    '''
    This function is used to get edge index and edge attribute for graph
    Args:
        records: pandas dataframe contains data each graph
        is_global: bool, if True is global graph (graph have all nodes), else is local graph (graph have only nodes in the same location)
    Return:
        edge_index: numpy array, shape (num_edges, 2), edge index of graph
        edge_attr: numpy array, shape (num_edges, 3), edge attribute of graph
    '''
    numRecords = len(records)
    if isGlobal:
        Long = records['Long'].values
        Lat = records['Lat'].values
    time = records['time'].values
    infectionPathways = records['infection_pathway'].values
    pathwayNames, pathwayCounts = np.unique(
        infectionPathways, return_counts=True)
    pathwayRatios = pathwayCounts / pathwayCounts.sum()
    pathwayRatioMap = dict(zip(pathwayNames, pathwayRatios))
    infectionPathways = records['infection_pathway'].apply(
        lambda x: pathwayRatioMap[x]).values

    rowIndices, colIndices = np.indices((numRecords, numRecords))
    mask = rowIndices != colIndices
    rowIndices = rowIndices[mask]
    colIndices = colIndices[mask]
    if isGlobal:
        longDiff = Long[rowIndices] - Long[colIndices]
        latDiff = Lat[rowIndices] - Lat[colIndices]
        distance = np.sqrt(longDiff**2 + latDiff**2)
        maxDistance = distance.max()
        scaledDistance = 1 - distance / maxDistance
    timeDiff = time[rowIndices] - time[colIndices]
    timeDiff = timeDiff / (timeDiff.max()-timeDiff.min())
    timeDiff[timeDiff < 0] = 0
    timeDiff = 1 - timeDiff
    isSamePathway = (
        infectionPathways[rowIndices] == infectionPathways[colIndices])

    epidemiologicalFactor = infectionPathways[rowIndices]
    epidemiologicalFactor[isSamePathway] = 1
    if isGlobal:
        hasEdge = scaledDistance * isSamePathway * timeDiff > 0.5
    else:
        hasEdge = isSamePathway * timeDiff > 0.5
    rowIndices = rowIndices[hasEdge]
    colIndices = colIndices[hasEdge]
    timeDiff = timeDiff[hasEdge]
    if isGlobal:
        longDiff = longDiff[hasEdge]
        latDiff = latDiff[hasEdge]
        edgeAttributes = np.column_stack((longDiff, latDiff, timeDiff))
    else:
        edgeAttributes = timeDiff.reshape(-1, 1)
    edgeIndex = np.column_stack((rowIndices, colIndices))
    return edgeIndex, edgeAttributes


class Datasets(InMemoryDataset):
    def __init__(self,
                 rootDir: str = './data',
                 mode: str = 'train',
                 datasetName: str = 'data',
                 targetDir: str = 'Preprocessed',
                 timeOffset: int = 3,
                 offsetType: str = 'month',
                 predictOffset: int = 1,
                 transform=None,
                 applyScaling=False,
                 pre_transform=None) -> None:
        super(Datasets, self).__init__(rootDir, transform, pre_transform)
        self.datasetName = datasetName
        self.root = rootDir
        self.offset = timeOffset
        self.targetDir = os.path.join(rootDir, targetDir)
        self.offsetType = offsetType
        self.predictOffset = predictOffset
        if not os.path.exists(self.targetDir):
            os.makedirs(self.targetDir)
        self.dataset = datasetName
        self.mode = mode
        self.applyScaling = applyScaling
        self.trainFileName = datasetName + "_train"
        self.testFileName = datasetName + "_test"
        self.ageGroups = [[0, 4], [5, 9], [10, 14], [15, 19], [20, 24], [
            25, 29], [30, 34], [35, 39], [40, 44], [45, 49], [50, 100]]
        self.process()

        if self.mode == "train":
            self.dataGraph = torch.load(os.path.join(
                self.targetDir, self.trainFileName + ".pt"))
        elif self.mode == "test":
            self.dataGraph = torch.load(os.path.join(
                self.targetDir, self.testFileName + ".pt"))
        else:
            self.dataGraph = torch.load(os.path.join(
                self.targetDir, self.datasetName + ".pt"))

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def getAgeGrp(self, age: int) -> int:
        for idx, grp in enumerate(self.ageGroups):
            if age >= grp[0] and age <= grp[1]:
                return idx
        return 6

    def preprocessFile(self) -> None:
        if os.path.exists(os.path.join(self.targetDir, self.datasetName + ".json")) and os.path.exists(os.path.join(self.targetDir, self.datasetName + ".csv")):
            self.infectiousObject = pickle.load(open(os.path.join(
                self.targetDir, self.datasetName + "_infectious_object.pkl"), 'rb'))
            self.occupation = pickle.load(
                open(os.path.join(self.targetDir, self.datasetName + "_occupation.pkl"), 'rb'))
            self.infectionPathway = pickle.load(
                open(os.path.join(self.targetDir, self.datasetName + "_infection_pathway.pkl"), 'rb'))
            self.sexs = pickle.load(
                open(os.path.join(self.targetDir, self.datasetName + "_sexs.pkl"), 'rb'))
            if os.path.exists(os.path.join(self.targetDir, self.datasetName + "_location_name.pkl")):
                self.locationName = pickle.load(
                    open(os.path.join(self.targetDir, self.datasetName + "_location_name.pkl"), 'rb'))
            if self.applyScaling:
                self.scaler = pickle.load(
                    open(os.path.join(self.targetDir, self.datasetName + "_scaler.pkl"), 'rb'))
            return

        data = pd.read_csv(os.path.join(self.root, self.datasetName + '.csv'))
        data.fillna("0", inplace=True)
        data['age_grp'] = data['age'].apply(self.getAgeGrp)
        self.locationName = list(data['district'].unique())
        with open(os.path.join(self.targetDir, self.datasetName + "_location_name.pkl"), 'wb') as f:
            pickle.dump(self.locationName, f)

        self.scaler = MinMaxScaler()
        if self.applyScaling:
            if os.path.exists(os.path.join(self.root, self.datasetName + "_scaler.pkl")):
                with open(os.path.join(self.root, self.datasetName + "_scaler.pkl"), 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                self.scaler.fit(data['new_case'].values.reshape(-1, 1))
                with open(os.path.join(self.targetDir, self.datasetName + "_scaler.pkl"), 'wb') as f:
                    pickle.dump(self.scaler, f)

        self.infectiousObject = list(data['infectious_object'].unique())
        with open(os.path.join(self.targetDir, self.datasetName + "_infectious_object.pkl"), 'wb') as f:
            pickle.dump(self.infectiousObject, f)
        self.occupation = list(data['occupation'].unique())
        with open(os.path.join(self.targetDir, self.datasetName + "_occupation.pkl"), 'wb') as f:
            pickle.dump(self.occupation, f)
        self.infectionPathway = list(data['infection_pathway'].unique())
        with open(os.path.join(self.targetDir, self.datasetName + "_infection_pathway.pkl"), 'wb') as f:
            pickle.dump(self.infectionPathway, f)
        self.sexs = list(data['sex'].unique())
        with open(os.path.join(self.targetDir, self.datasetName + "_sexs.pkl"), 'wb') as f:
            pickle.dump(self.sexs, f)
        data['date_of_infection'] = pd.to_datetime(
            data['date_of_infection'], format='mixed')
        data['month-year'] = pd.to_datetime(data['month-year'], format='mixed')
        if self.offsetType == 'month':
            minDate = data['month-year'].min()
            currentDate = minDate
            maxDate = data['month-year'].max()
            indexData = []
            while currentDate <= maxDate - pd.DateOffset(months=self.offset):
                border = currentDate + pd.DateOffset(months=self.offset)
                current_date_str = currentDate.strftime('%Y-%m-%d')
                border_str = border.strftime('%Y-%m-%d')
                subData = data.query(
                    '`month-year` >= @current_date_str and `month-year` < @border_str')
                labelData = data.query('`month-year` == @border_str')
                indexData.append((list(subData.index.values.astype('float')), list(
                    labelData.index.values.astype('float'))))
                currentDate = currentDate + \
                    pd.DateOffset(months=self.predictOffset)
        else:
            minDate = data['date_of_infection'].min()
            currentDate = minDate
            maxDate = data['date_of_infection'].max()
            indexData = []
            while currentDate <= maxDate - pd.DateOffset(days=self.offset):
                border = currentDate + pd.DateOffset(days=self.offset)
                current_date_str = currentDate.strftime('%Y-%m-%d')
                border_str = border.strftime('%Y-%m-%d')
                subData = data.query(
                    '`ngaynhiemhiv` >= @current_date_str and `ngaynhiemhiv` < @border_str')
                borderNew = border + pd.DateOffset(days=self.predictOffset)
                borderNew = borderNew.strftime('%Y-%m-%d')
                labelData = data.query(
                    '`ngaynhiemhiv` >= @border_str and `ngaynhiemhiv` < @borderNew')
                indexData.append((list(subData.index.values.astype('float')), list(
                    labelData.index.values.astype('float'))))
                currentDate = currentDate + \
                    pd.DateOffset(days=self.predictOffset)

        indexRecords = []
        for index in indexData:
            tmp = data.iloc[index[1]].groupby('district')
            label = []
            for location in self.locationName:
                if location in tmp.groups:
                    label.append(tmp.get_group(location)[
                                 'new_case'].values[0]*1.0)
                else:
                    label.append(0)
            if self.applyScaling:
                label = self.scaler.transform(
                    np.array(label).reshape(-1, 1)).tolist()
            indexRecords.append((index[0], label))
        self.data = data
        with open(os.path.join(self.targetDir, self.datasetName + ".json"), 'w') as f:
            json.dump(indexRecords, f)
        data.to_csv(os.path.join(
            self.targetDir, self.datasetName + ".csv"), index=False)
        print(f"Preprocess file {self.datasetName} done")
        return

    def process(self) -> None:
        self.preprocessFile()
        if self.__isExistFiles(self.targetDir, self.testFileName, self.trainFileName, ".pt"):
            return

        self.__prePreprocessingData(
            self.datasetName, f'Convert data train to torch format')

        return

    def __prePreprocessingData(self, fileName: str, message: str) -> None:
        """
            Save data after collating
        """

        data = pd.read_csv(os.path.join(self.targetDir, fileName + ".csv"))
        indexRecords = json.load(
            open(os.path.join(self.targetDir, self.datasetName + ".json")))
        listData = []

        for idx in tqdm(range(len(indexRecords)), desc=f"Create data {fileName}"):
            index, label = indexRecords[idx]
            records = data.iloc[index]
            # nodes features
            ageGrp = records['age_grp'].values.tolist()
            infectiouObjects = records['infectious_object'].apply(
                lambda x: self.infectiousObject.index(x)).values.tolist()
            occupation = records['occupation'].apply(
                lambda x: self.occupation.index(x)).values.tolist()
            infectionPathways = records['infection_pathway'].apply(
                lambda x: self.infectionPathway.index(x)).values.tolist()
            sex = records['sex'].apply(
                lambda x: self.sexs.index(x)).values.tolist()
            physPosition = records[['Lat', 'Long']].values.tolist()
            time = records['time'].values.tolist()
            newCase = records['new_case'].values.tolist()
            if self.applyScaling:
                newCase = self.scaler.transform(
                    np.array(newCase).reshape(-1, 1)).tolist()
            batchLocal = records['district'].apply(
                lambda x: self.locationName.index(x)).values.tolist()
            edgeIndex, edgeAttr = getEdges(records)
            groupDistrict = records.groupby('district')
            group0 = groupDistrict.get_group(self.locationName[0])
            edgeIndexLocals, edgeAttrLocals = getEdges(
                group0, isGlobal=False)
            currentNumNode = len(group0)
            for district in self.locationName[1:]:
                group = groupDistrict.get_group(district)
                edgeIndexLocal, edgeAttrLocal = getEdges(
                    group, isGlobal=False)
                edgeIndexLocals = np.vstack(
                    (edgeIndexLocals, edgeIndexLocal+currentNumNode))
                edgeAttrLocals = np.vstack(
                    (edgeAttrLocals, edgeAttrLocal))
                currentNumNode += len(group)
            # rr = torch.LongTensor(edge_index_locals)
            graphData = gemetric_data.Data(x=None,
                                            nodes={
                                                "infectious_object": torch.LongTensor(infectiouObjects),
                                                "occupation": torch.LongTensor(occupation),
                                                "infection_route": torch.LongTensor(infectionPathways),
                                                "sex": torch.LongTensor(sex),
                                                "phys_pos": torch.FloatTensor(physPosition),
                                                "time": torch.FloatTensor(time),
                                                "new_case": torch.FloatTensor(newCase),
                                                "age_grp": torch.LongTensor(ageGrp),
                                                "location": torch.LongTensor(batchLocal),
                                                "num_edge_local": torch.LongTensor([len(edgeIndexLocals)]),
                                                "location_edge": torch.LongTensor(edgeIndexLocals),
                                                "location_edge_attr": torch.FloatTensor(edgeAttrLocals)
                                            },
                                            edge_index=torch.LongTensor(
                                                edgeIndex).transpose(1, 0),
                                            edge_attr=torch.FloatTensor(
                                                edgeAttr),
                                            y=torch.FloatTensor(label))
            listData.append(graphData)
        torch.save(listData, os.path.join(
            self.targetDir, self.datasetName + ".pt"))
        data_train, data_test = train_test_split(
            listData, test_size=0.2, random_state=42)
        torch.save(data_train, os.path.join(
            self.targetDir, self.trainFileName + ".pt"))
        torch.save(data_test, os.path.join(
            self.targetDir, self.testFileName + ".pt"))
        print(f"{message} done")
        return

    def __len__(self):
        return len(self.dataGraph)

    def __getitem__(self, idx):
        return self.dataGraph[idx]

    @staticmethod
    def __isExistFiles(root: str,
                       fileNameTest: str,
                       filesNameTrain: str,
                       format: str) -> bool:
        """ 
            Check that valid input dataset files to run the model 
        """
        fileNameTest = os.path.join(root, fileNameTest)
        filesNameTrain = os.path.join(root, filesNameTrain)

        if os.path.exists(fileNameTest + format) and os.path.exists(filesNameTrain + format):
            return True

        return False
