import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
              "open-images-v7",
              split="validation",
              label_types=["detections"],
              classes=["Bottle"],
              max_samples=350,
          )

session = fo.launch_app(dataset)