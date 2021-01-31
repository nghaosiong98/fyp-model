from model import AlgaeClassificationModel
from dataset import Dataset
from utils import save_loss_graph, save_accuracy_graph, make_dir, save_confussion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--annotation', help='path to csv annotation file', required=True)
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--output', help='path to keep log, figures, and checkpoints')
args = parser.parse_args()

model = AlgaeClassificationModel().model
dataset = Dataset(args.annotation, args.data)
output_path = args.output
checkpoint_path = os.path.join(output_path, 'checkpoints')


def train():
    hist = model.fit(
        x=dataset.generator(data_type='train', flip=True, rotate=True),
        epochs=20,
        steps_per_epoch=10,
        class_weight=dataset.get_class_weight(),
        validation_data=dataset.generator(data_type='val')
    )
    save_loss_graph(hist, output_path)
    save_accuracy_graph(hist, output_path)
    make_dir(checkpoint_path)
    model.save(checkpoint_path)


def predict():
    pred = model.predict_generator(
        dataset.generator(data_type='test'),
        steps=len(dataset.test),
        verbose=1,
    )
    predicted_class_indices = np.argmax(pred, axis=1)
    print(predicted_class_indices)
    test_preds = np.argmax(pred, axis=1)
    test_trues = np.argmax(dataset.test_y, axis=-1)
    cm = confusion_matrix(test_trues, test_preds)
    save_confussion_matrix(cm, index=['With algae', 'Without algae'], columns=['With algae', 'Without algae'])
    print(classification_report(test_trues, test_preds))
