import torch
import torch.nn as nn
import numpy as np

from scipy.stats import pearsonr

from utils import rank_array
from candidate_sampling import candidate_subsampling
from model_moe import MoE, MLPTower, MLPExpert



class ModelMultitaskBinary(nn.Module):

    def __init__(self, pretrained_model, tokenizer, args):
        super(ModelMultitaskBinary, self).__init__()
        self.tokenizer = tokenizer
        self.args = args

        # LM
        self.pretrained_model = pretrained_model
        # shared bottom
        self.fc1 = nn.Linear(args.hidden_size, args.bottom_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.bottom_hidden_size, args.hidden_size)
        # MoE
        self.moe = MoE(args.device, args.n_tasks, args.hidden_size, args.hidden_size, args.num_experts, args.expert_hidden_size, args.k)
        # towers - one for each task
        self.towers = nn.ModuleList([MLPTower(args.hidden_size, args.tower_hidden_size) for i in range(args.n_tasks)])
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCEWithLogitsLoss()

        # sampled candidates
        self.selected_idx = []

        # training labels
        self.original_training_labels = {}
        self.training_labels = {}
        self.training_scores = {}
        self.training_hits = {}
        for j in range(self.args.n_tasks):
            self.original_training_labels[j] = []
            self.training_labels[j] = []
            self.training_scores[j] = []
            self.training_hits[j] = []

        # multi-summary evaluation
        self.multi_summary_pred_idx = {}
        self.multi_summary_preds = {}
        for j in range(self.args.n_tasks):
            self.multi_summary_pred_idx[j] = []
            self.multi_summary_preds[j] = []

    def display_selected_idx(self):
        print("\nStatistics on sampled candidates:")
        n_methods = len(self.args.generation_methods)
        selected_methods = {}
        for i in range(len(self.selected_idx)):
            idx = self.selected_idx[i]
            method = int(idx / self.args.num_beams)
            if not(method in selected_methods.keys()):
                selected_methods[method] = 0
            selected_methods[method] += 1
        for method in selected_methods.keys():
            print("Generation method {}, # selected candidates: {} ({:.4f}%)".format(
                method, selected_methods[method], 100 * selected_methods[method] / len(self.selected_idx)
            ))

    def display_training_labels(self):
        print("\nStatistics on training labels:")
        for j in range(self.args.n_tasks):
            s_ori_pos_j = np.sum(self.original_training_labels[j])
            s_pos_j = np.sum(self.training_labels[j])
            m_pos_j = 100 * np.mean(self.training_labels[j]) / (self.args.n_positives + self.args.n_negatives)
            m_label_j = np.mean(self.training_scores[j])
            m_hits_j = 100 * np.mean(self.training_hits[j])
            s_hits_j = np.sum(self.training_hits[j])
            print("Task {}, # original pos: {} / {} batches // # pos: {} / {} batches, % pos: {:.4f} // mean of training label: {:.4f} // % hitting the max: {:.4f}, count: {} / {}".format(
                j, s_ori_pos_j, len(self.training_labels[j]),  s_pos_j, len(self.training_labels[j]), m_pos_j, m_label_j, m_hits_j, s_hits_j, len(self.training_hits[j])
            ))

    def display_multi_summary_predictions(self):
        print("\nMulti-summary evaluation:")
        all_ms = []
        for j in range(self.args.n_tasks):
            self.multi_summary_pred_idx[j] = np.array(self.multi_summary_pred_idx[j])
            self.multi_summary_preds[j] = np.array(self.multi_summary_preds[j])
            m_j = np.mean(self.multi_summary_preds[j])
            all_ms.append(m_j)
            print("Task {}, prediction is {:.4f}".format(j, m_j))
        print("Mean over tasks: {:.4f}".format(np.mean(all_ms)))
        intersections = []
        correlations = []
        for j in range(self.args.n_tasks):
            for k in range(self.args.n_tasks):
                if k != j:
                    intersect = 100 * np.mean(self.multi_summary_pred_idx[j] == self.multi_summary_pred_idx[k])
                    intersections.append(intersect)
                    corr, p = pearsonr(self.multi_summary_pred_idx[j], self.multi_summary_pred_idx[k])
                    correlations.append(corr)
        m_intersection = np.mean(intersections)
        m_corr = np.mean(correlations)
        print("Mean intersection between pairs of pred idx: {:.4f}, mean Pearson correlation: {:.4f}".format(m_intersection, m_corr))
            
    def forward(self, mode, text_and_summaries_ids, text_and_summaries_mask, scores):
        loss = torch.tensor(0.0).to(self.pretrained_model.device)
        accuracy = [0 for j in range(self.args.n_tasks)]
        rank = [0 for j in range(self.args.n_tasks)]
        predictions_idx = [[] for j in range(self.args.n_tasks)]
        predictions = [[] for j in range(self.args.n_tasks)]
        total_predictions_idx = []
        overall_sums = []
        overall_predictions = []
        for i in range(text_and_summaries_ids.shape[0]):

            # data
            text_and_summaries_ids_i = text_and_summaries_ids[i]
            text_and_summaries_mask_i = text_and_summaries_mask[i]

            # labels construction
            scores_i = scores[i]
            original_scores_i = scores_i.clone().detach()
            labels_i = torch.zeros(scores_i.shape, device = self.pretrained_model.device)
            for j in range(self.args.n_tasks):
                best_j = scores_i[j].max()
                if self.args.sharp_pos:
                    if best_j > scores_i[j].min():
                        labels_i[j][scores_i[j] == best_j] = 1
                else:
                    labels_i[j][scores_i[j] == best_j] = 1
            original_labels_i = labels_i.clone().detach()

            # candidate sampling
            selected_idx, text_and_summaries_ids_i, text_and_summaries_mask_i, scores_i, labels_i = candidate_subsampling(
                mode, text_and_summaries_ids_i, text_and_summaries_mask_i, scores_i, labels_i, self.args
            )
            self.selected_idx += selected_idx

            # model output
            # LM encoding
            outputs_i = self.pretrained_model(
                input_ids = text_and_summaries_ids_i, attention_mask = text_and_summaries_mask_i, output_hidden_states = True
            )
            encs = outputs_i["last_hidden_state"]
            encs = encs[:, 0, :]
            # shared bottom
            if self.args.use_shared_bottom:
                preds_i = self.fc2(self.relu(self.fc1(encs)))
            else:
                preds_i = encs
            # MoE
            train = torch.sum(mode) > 0
            preds_i, aux_loss_i = self.moe(preds_i, train = train, collect_gates = not(train))
    
            loss_i = torch.tensor(0.0).to(self.pretrained_model.device)
            total_predictions = np.zeros(len(preds_i[0]))
            for j in range(self.args.n_tasks):

                # pred
                preds_i_j = self.towers[j](preds_i[j])[:, 0]

                # labels
                labels_i_j = labels_i[j]
                if torch.sum(mode) > 0:
                    self.original_training_labels[j].append(original_labels_i[j].sum().item())
                    self.training_labels[j].append(labels_i_j.sum().item())
                    if labels_i_j.sum() > 0:
                        self.training_scores[j].append(scores_i[j][labels_i_j == 1].mean().item())
                    self.training_hits[j].append(int(scores_i[j].max().item() == original_scores_i[j].max().item()))

                # loss
                loss_i_j = self.loss(preds_i_j, labels_i_j)
                loss_i = loss_i + loss_i_j

                # predictions
                preds_i_j = self.sigmoid(preds_i_j).detach().cpu().numpy()
                prediction_idx = np.argmax(preds_i_j)
                predictions_idx[j].append(prediction_idx)
                prediction = scores_i[j][prediction_idx].item()
                predictions[j].append(prediction)
                total_predictions += preds_i_j

                # accuracy
                pos_idx = scores_i[j].argmax().item()
                accuracy_i_j = 100 * int(scores_i[j][prediction_idx].item() == scores_i[j][pos_idx].item())
                accuracy[j] = accuracy[j] + accuracy_i_j

                # ranks
                ranks = rank_array(preds_i_j)
                all_pos_idx = [k for k in range(len(scores_i[j])) if scores_i[j][k].item() == scores_i[j][pos_idx].item()]
                rank_i_j = np.min(ranks[all_pos_idx])
                rank[j] = rank[j] + rank_i_j
            loss_i = loss_i / self.args.n_tasks
            if self.args.use_aux_loss:
                loss_i = loss_i + aux_loss_i
            loss = loss + loss_i
            total_predictions /= self.args.n_tasks
            total_prediction_idx = np.argmax(total_predictions)
            total_predictions_idx.append(total_prediction_idx)
            overall_sum = sum([scores_i[j][total_prediction_idx].item() for j in range(self.args.n_tasks)])
            overall_sums.append(overall_sum)
            overall_predictions.append(total_predictions)

        loss /= scores.shape[0]
        outputs = {
            "loss": loss,
            "loss_nce": loss,
            "total_predictions_idx": total_predictions_idx,
            "overall_predictions": overall_predictions
        }
        prediction_sum = 0
        for j in range(self.args.n_tasks):
            accuracy[j] /= scores.shape[0]
            outputs["accuracy_{}".format(self.args.scoring_methods[j])] = torch.tensor(accuracy[j]).float().to(loss.device)
            rank[j] /= scores.shape[0]
            outputs["rank_{}".format(self.args.scoring_methods[j])] = torch.tensor(rank[j]).float().to(loss.device)
            if torch.sum(mode) == 0:
                self.multi_summary_pred_idx[j] += predictions_idx[j]
                self.multi_summary_preds[j] += predictions[j]
            predictions[j] = np.mean(predictions[j])
            outputs["prediction_{}".format(self.args.scoring_methods[j])] = torch.tensor(predictions[j]).float().to(loss.device)
            prediction_sum += predictions[j]
        outputs["prediction_sum"] = torch.tensor(prediction_sum).float().to(loss.device)
        outputs["overall_sum"] = torch.tensor(np.mean(overall_sums)).float().to(loss.device)
        
        return outputs


