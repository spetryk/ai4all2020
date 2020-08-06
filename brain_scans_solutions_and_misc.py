class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(25088, 1),
                nn.Sigmoid())

    def forward(self, x):
        return self.model(x.unsqueeze(1))model = MyModel()

(torch.round(predictions.view(-1)) == labels).sum()
calculate_loss(predictions, labels)


softmax = nn.Softmax()
num_wrong = 0
for i in range(len(val_dataset)):
    img, label = val_dataset[i]
    prediction = model(img.unsqueeze(0)).item()
    if round(prediction) != label and label == 1:
        num_wrong += 1
        print("i", i)
        print("label:", label)
        print("prediction:", prediction)
print(num_wrong, len(val_dataset))
#torch.save(model.state_dict(), "trained_linear_model.pt")



img, label = train_dataset[1]
print("label: {} ({})".format(label, "real" if label else "fake"))
plt.imshow(img, interpolation="bicubic")

img, label = train_dataset[10]
print("label: {} ({})".format(label, "real" if label else "fake"))
plt.figure()
plt.imshow(img, interpolation="bicubic")

img, label = val_dataset[158]
print("label: {} ({})".format(label, "real" if label else "fake"))
plt.figure()
plt.imshow(img, interpolation="bicubic")
