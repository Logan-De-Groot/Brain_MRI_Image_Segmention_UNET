from unet import *
import csv
import pathlib
import sys


TRAIN_PATH = "keras_png_slices_data/keras_png_slices_train"
TRAIN_SEG_PATH = "keras_png_slices_data/keras_png_slices_seg_train"
VAL_PATH = "keras_png_slices_data/keras_png_slices_validate"
VAL_SEG_PATH = "keras_png_slices_data/keras_png_slices_seg_validate"
TEST_PATH = "keras_png_slices_data/keras_png_slices_test"
TEST_SEG_PATH = "keras_png_slices_data/keras_png_slices_seg_test"

BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
NUM_WORKERS = 12

def train(data, optimizer, model, criterion, running_loss, train=True):
    """Load the given data and train the model and calculate loss. Skips training if validation is passed"""
    x, y = data
    x = x.to(device, dtype = torch.float32)
    y = y.to(device, dtype = torch.long)

    if train:
        x.requires_grad = True
        optimizer.zero_grad()

    outputs = model(x).cuda()
    loss = criterion(outputs, y.squeeze(1)).to(device)

    if train:
        loss.backward()
        optimizer.step()
    running_loss += loss.item()

    return running_loss


def train_model():
    """
    Trains the model and saves the model off the specified path to files
    """
    # Load Data
    train_slices_loader = ImageLoader(TRAIN_PATH,TRAIN_SEG_PATH)
    train_slices = DataLoader(train_slices_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers= NUM_WORKERS)

    validate_slices_loader = ImageLoader(VAL_PATH,VAL_SEG_PATH)
    validate_slices = DataLoader(validate_slices_loader, batch_size=BATCH_SIZE, shuffle=True, num_workers= 12)

    # Load params and network
    model = MainNetwork()
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    running_lost_all = []
    validation_lost_all = []
    # Loop through each epoch training on data
    for epoch in range(EPOCHS):

        #calculate training set
        start_time = time.time()

        running_loss = 0
        model.train(True)
        for index, data in enumerate(tqdm(train_slices)):
            running_loss = train(data, optimizer, model, criterion, running_loss)

        print("Training Loss is:", running_loss / len(train_slices))
        running_lost_all.append(running_loss / len(train_slices))
        running_loss = 0
        model.train(False)

        # calculate validation set
        with torch.no_grad():
            for index, data in enumerate(validate_slices):
                running_loss = train(data, optimizer, model, criterion, running_loss, train=False)

        print("Validation Loss is:", running_loss / len(validate_slices))
        validation_lost_all.append(running_loss / len(validate_slices))
        print("Finished Epoch in: ", time.time() - start_time)
        print("Epochs left: ", EPOCHS - epoch - 1)

    torch.save(model, "Segmentation_Model")


def show_comparision(model):
    """
    Show a comparison using matplotlib based off the test set of data
    :param model:  the trained model unevaluated
    :return: nothing
    """
    model.eval()

    test_slices_loader = ImageLoader(TEST_PATH,TEST_SEG_PATH)

    test_slices = DataLoader(test_slices_loader, batch_size=1, shuffle=True)
    fig = plt.figure(figsize=(14, 7))

    for index, data in enumerate(test_slices):
        x, y = data

        output_image = model.forward(x.cuda())
        output_image = output_image[0]

        output_image = torch.argmax(output_image, 0)

        fig.add_subplot(1, 3, 1)
        plt.imshow(x[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.title("Test Image Non Segmented (X)")
        fig.add_subplot(1, 3, 2)
        plt.imshow(y[0].permute(1, 2, 0).detach().cpu().numpy())
        plt.title("Reference Test Image Segmented (Y)")
        fig.add_subplot(1, 3, 3)
        plt.imshow(output_image.detach().cpu().numpy())
        plt.title("Generated Segementation Map")
        break
    plt.show()


if __name__ == '__main__':
    # try load model based off standard name else ask for path
    if len(sys.argv) > 1:
        if sys.argv[1] == 'show':
            try:
                model = torch.load(os.path.join(pathlib.Path(__file__).parent.resolve(), "Segmentation_Model"))
            except Exception as e:
                path = input("Failed to load default model, specify model path: ")
                try:
                    model = torch.load(path)
                except Exception:
                    print("Failed to load model, verify path and try again")
                    model = False

            if model is not False:
                show_comparision(model)
    else:
        train_model()
