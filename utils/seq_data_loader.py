from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence


class LoanSimDataset(Dataset):
    def __init__(
        self,
        csv_file_path,
        scale_columns,
        feature_columns,
        target_columns,
        group="train",
    ):
        # -------------------------------------------------------
        # read the csv file
        data = pd.read_csv(csv_file_path)
        # choose train or test data for RLsimulator
        # rlsim_data = data.loc[(data["group"] == group)]
        rlsim_train_data = data.loc[data["group"] == "train"]
        rlsim_test_data = data.loc[data["group"] == "test"]
        # -------------------------------------------------------
        # scale the data
        scaler = StandardScaler()
        # we use train data to fit the scaler
        train_scaled = scaler.fit_transform(rlsim_train_data[scale_columns])
        if group == "train":
            # rlsim_data = rlsim_train_data
            train_scaled_df = pd.DataFrame(train_scaled, columns=scale_columns)
            train_scaled_df_full = pd.concat(
                [
                    train_scaled_df.reset_index(drop=True),
                    rlsim_train_data[
                        [
                            "installment_done",
                            "loan_done",
                            "recovery_rate_weighted",
                            "loan_id",
                        ]
                    ].reset_index(drop=True),
                ],
                axis=1,
            )
            rlsim_data = train_scaled_df_full
        else:
            test_scaled = scaler.transform(rlsim_test_data[scale_columns])
            test_scaled_df = pd.DataFrame(test_scaled, columns=scale_columns)
            test_scaled_df_full = pd.concat(
                [
                    test_scaled_df.reset_index(drop=True),
                    rlsim_test_data[
                        [
                            "installment_done",
                            "loan_done",
                            "recovery_rate_weighted",
                            "loan_id",
                        ]
                    ].reset_index(drop=True),
                ],
                axis=1,
            )
            rlsim_data = test_scaled_df_full
        # print(rlsim_data.head(14))
        # -------------------------------------------------------
        # group the data by loan_id
        self.grouped_data = rlsim_data.groupby("loan_id")
        # -------------------------------------------------------
        self.features_columns = feature_columns
        self.target_columns = target_columns
        self.sequences = self.create_sequences()

    def create_sequences(self):
        print("------------ Creating Sequences ------------")
        sequences = []
        for loan_id, group in self.grouped_data:
            print(loan_id)
            loan_id: str = loan_id
            features = group[self.features_columns].values
            targets = group[self.target_columns].values
            sequences.append((features, targets))
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, target = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )


def collate_fn(batch):
    # unzip the batch and get the features and targets
    features, targets = zip(*batch)
    # using pad_sequence to pad sequences to the same length
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    # get the actual lengths of the sequences
    lengths = torch.tensor([len(f) for f in features])
    return padded_features, padded_targets, lengths
