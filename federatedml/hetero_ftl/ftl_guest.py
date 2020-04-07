from federatedml.hetero_ftl.ftl_base import FTL
from federatedml.statistic.intersect import intersect_guest
from arch.api.utils import log_utils
from federatedml.hetero_ftl.ftl_dataloder import FTLDataLoader

from federatedml.nn.hetero_nn.backend.paillier_tensor import PaillierTensor

import numpy as np

from federatedml.util import consts

LOGGER = log_utils.getLogger()


class FTLGuest(FTL):

    def __init__(self):
        super(FTLGuest, self).__init__()
        self.phi = None  # Φ_A
        self.phi_product = None  # (Φ_A)‘(Φ_A) [feature_dim, feature_dim]
        self.overlap_y = None  # y_i ∈ N_c
        self.overlap_y_2 = None  # (y_i ∈ N_c )^2
        self.overlap_ua = None  # u_i ∈ N_AB
        self.constant_k = None  # κ
        self.feat_dim = None  # output feature dimension

        self.overlap_y_pt = None  # paillier tensor

        self.role = consts.GUEST

    def init_intersect_obj(self):
        intersect_obj = intersect_guest.RsaIntersectionGuest(self.intersect_param)
        intersect_obj.guest_party_id = self.component_properties.local_partyid
        intersect_obj.host_party_id_list = self.component_properties.host_party_idlist
        LOGGER.debug('intersect done')
        return intersect_obj

    def compute_phi_and_overlap_ua(self, data_loader: FTLDataLoader):
        phi = None  # [1, feature_dim]  Φ_A
        overlap_ua = []
        for i in range(len(data_loader)):
            batch_x, batch_y = data_loader[i]
            ua_batch = self.nn.predict(batch_x)  # [batch_size, feature_dim]

            relative_overlap_index = data_loader.get_relative_overlap_index(i)
            if len(relative_overlap_index) != 0:
                if self.verbose:
                    LOGGER.debug('batch {}/{} overlap index is {}'.format(i, len(data_loader), relative_overlap_index))
                overlap_ua.append(ua_batch[relative_overlap_index])

            phi_tmp = np.expand_dims(np.sum(batch_y * ua_batch, axis=0), axis=0)
            if phi is None:
                phi = phi_tmp
            else:
                phi += phi_tmp

        return phi, overlap_ua

    def batch_compute_components(self, data_loader: FTLDataLoader, epoch_idx):

        phi, overlap_ua = self.compute_phi_and_overlap_ua(data_loader)

        phi = phi / self.data_num  # Φ_A [1, feature_dim]
        phi_product = np.matmul(phi.transpose(), phi)  # (Φ_A)‘(Φ_A) [feature_dim, feature_dim]

        overlap_y = data_loader.get_overlap_y()  # {C(y)=y} [1, feat_dim]
        overlap_y_2 = overlap_y * overlap_y  # {D(y)=y^2} # [1, feat_dim]

        overlap_ua = np.concatenate(overlap_ua, axis=0)  # [overlap_num, feat_dim]

        # if self.verbose:
        #     LOGGER.debug('epoch idx is {}'.format(epoch_idx))
        #     LOGGER.debug('phi_product is {}'.format(phi_product))
        #     LOGGER.debug('phi is {}'.format(phi))
        #     LOGGER.debug('overlap_ua is {}'.format(overlap_ua))
        #     LOGGER.debug('overlap_y {}, overlap_y2 {}'.format(overlap_y, overlap_y_2))

        # 3 components send to host
        y_overlap_2_phi_2 = 0.25 * np.expand_dims(overlap_y_2, axis=2) * phi_product
        y_overlap_phi = -0.5 * overlap_y * phi
        mapping_comp_a = - overlap_ua * self.constant_k

        return phi, phi_product, overlap_y, overlap_y_2, overlap_ua, [y_overlap_2_phi_2, y_overlap_phi, mapping_comp_a]

    def exchange_components(self, comp_to_send, epoch_idx):

        if self.mode == 'encrypted':
            comp_to_send = self.encrypt_tensor(comp_to_send)

        self.transfer_variable.guest_components.remote(comp_to_send, suffix=(epoch_idx, 'exchange_guest_comp'))
        host_components = self.transfer_variable.host_components.get(idx=0, suffix=(epoch_idx, 'exchange_host_comp'))
        return host_components

    def decrypt_inter_result(self, encrypted_const, grad_a_overlap, epoch_idx):

        rand_0 = self.rng_generator.generate_random_number(encrypted_const.shape)
        encrypted_const = encrypted_const + rand_0
        rand_1 = PaillierTensor(ori_data=self.rng_generator.generate_random_number(grad_a_overlap.shape))
        grad_a_overlap = grad_a_overlap + rand_1

        send_data = [encrypted_const, grad_a_overlap]
        self.transfer_variable.guest_side_gradients.remote(send_data, suffix=(epoch_idx, 'guest_de_send'))
        rs = self.transfer_variable.decrypted_guest_gradients.get(suffix=(epoch_idx, 'guest_de_get'), idx=0)

        const = rs[0] - rand_0
        grad_a_overlap = rs[1] - rand_1

        return const, grad_a_overlap

    def decrypt_host_data(self, epoch_idx):

        inter_grad = self.transfer_variable.host_side_gradients.get(suffix=(epoch_idx, 'host_de_send'), idx=0)
        self.transfer_variable.decrypted_host_gradients.remote(inter_grad.decrypt(self.encrypter),
                                                               suffix=(epoch_idx, 'host_de_get'))

    def decrypt_loss_val(self, encrypted_loss, epoch_idx):
        self.transfer_variable.encrypted_loss.remote(encrypted_loss, suffix=(epoch_idx, 'send_loss'))
        decrypted_loss = self.transfer_variable.decrypted_loss.get(idx=0, suffix=(epoch_idx, 'get_loss'))
        return decrypted_loss

    def compute_backward_gradients(self, host_components, data_loader: FTLDataLoader, epoch_idx):

        # they are Paillier tensors or np array
        overlap_ub, overlap_ub_2, mapping_comp_b = host_components[0], host_components[1], host_components[2]

        y_overlap_2_phi = np.expand_dims(self.overlap_y_2 * self.phi, axis=1)

        if self.mode == 'plain':

            LOGGER.debug('run plain mode')

            loss_grads_const_part1 = 0.25 * np.squeeze(np.matmul(y_overlap_2_phi, overlap_ub_2), axis=1)
            loss_grads_const_part2 = self.overlap_y * overlap_ub

            const = np.sum(loss_grads_const_part1, axis=0) - 0.5 * np.sum(loss_grads_const_part2, axis=0)

            grad_a_nonoverlap = self.alpha * const * data_loader.y[data_loader.get_non_overlap_indexes()] / self.data_num
            grad_a_overlap = self.alpha * const * self.overlap_y / self.data_num + mapping_comp_b

            return np.concatenate([grad_a_overlap, grad_a_nonoverlap], axis=0)

        elif self.mode == 'encrypted':

            LOGGER.debug('run encrypted mode')

            loss_grads_const_part1 = overlap_ub_2.matmul_3d(0.25 * y_overlap_2_phi, multiply='right')
            loss_grads_const_part1 = loss_grads_const_part1.squeeze(axis=1)

            if self.overlap_y_pt is None:
                self.overlap_y_pt = PaillierTensor(self.overlap_y, partitions=self.partitions)

            loss_grads_const_part2 = overlap_ub * self.overlap_y_pt

            encrypted_const = loss_grads_const_part1.reduce_sum() - 0.5 * loss_grads_const_part2.reduce_sum()

            grad_a_overlap = self.overlap_y_pt.map_ndarray_product((self.alpha/self.data_num * encrypted_const)) + mapping_comp_b

            const, grad_a_overlap = self.decrypt_inter_result(encrypted_const, grad_a_overlap, epoch_idx=epoch_idx)

            self.decrypt_host_data(epoch_idx)

            grad_a_nonoverlap = self.alpha * const * data_loader.y[data_loader.get_non_overlap_indexes()]/self.data_num

            return np.concatenate([grad_a_overlap.numpy(), grad_a_nonoverlap], axis=0)

    def compute_loss(self, host_components, epoch_idx):

        overlap_ub, overlap_ub_2, mapping_comp_b = host_components[0], host_components[1], host_components[2]

        if self.mode == 'plain':

            # FIXME / feat_dim possible problem
            loss_overlap = np.sum((self.overlap_ua / self.feat_dim) * overlap_ub)

            ub_phi = np.matmul(overlap_ub, self.phi.transpose())
            loss_y = (-0.5*np.sum(self.overlap_y*ub_phi)+1.0/8*np.sum(ub_phi * ub_phi))+len(self.overlap_y)*np.log(2)

            return self.alpha * loss_y + loss_overlap

        elif self.mode == 'encrypted':

            loss_overlap = overlap_ub.element_wise_product((self.overlap_ua*self.constant_k))
            sum = np.sum(loss_overlap.reduce_sum())
            ub_phi = overlap_ub.T.fast_matmul_2d(self.phi.transpose())

            part1 = -0.5 * np.sum((self.overlap_y * ub_phi))
            ub_2 = overlap_ub_2.reduce_sum()
            enc_phi_uB_2_phi = np.matmul(np.matmul(self.phi, ub_2), self.phi.transpose())
            part2 = 1/8 * np.sum(enc_phi_uB_2_phi)
            part3 = len(self.overlap_y)*np.log(2)

            loss_y = part1 + part2 + part3
            en_loss = self.alpha * loss_y + sum
            loss_val = self.decrypt_loss_val(en_loss, epoch_idx)

            return loss_val

    def fit(self, data_inst, validate_data):

        data_loader, self.x_shape, self.data_num, self.overlap_num = self.prepare_data(self.init_intersect_obj(),
                                                                                       data_inst, guest_side=True)

        self.partitions = data_inst._partitions

        self.initialize_nn(input_shape=self.x_shape)
        self.feat_dim = self.nn._model.output_shape[1]
        self.constant_k = 1 / self.feat_dim

        LOGGER.debug('init weights are {}'.format(self.nn.get_trainable_weights()))

        for epoch_idx in range(self.epochs):

            LOGGER.debug('fitting epoch {}'.format(epoch_idx))

            self.phi, self.phi_product, self.overlap_y, self.overlap_y_2, self.overlap_ua, send_components = \
                self.batch_compute_components(data_loader, epoch_idx)

            host_components = self.exchange_components(send_components, epoch_idx=epoch_idx)

            grads = self.compute_backward_gradients(host_components, data_loader, epoch_idx=epoch_idx)

            self.update_nn_weights(grads, data_loader, epoch_idx)

            loss = self.compute_loss(host_components, epoch_idx)

            LOGGER.debug('fitting epoch {} done, loss is {}'.format(epoch_idx, loss))

        phi, overlap_ua = self.compute_phi_and_overlap_ua(data_loader)
        self.phi = phi

        LOGGER.debug('fitting ftl model done')

        # self.predict(None)

    def predict(self, data_inst):

        LOGGER.debug('guest start to predict')
        LOGGER.debug('phi shape is {}'.format(self.phi.shape))

        batch_num = self.transfer_variable.predict_batch_num.get(idx=0, suffix=('predict_batch_num', ))

        for i in range(batch_num):

            host_u = self.transfer_variable.predict_host_u.get(idx=0, suffix=(i, 'host_u'))
            en_scores = host_u.fast_matmul_2d(self.phi.transpose())
            rand = self.rng_generator.generate_random_number(en_scores.shape)
            self.transfer_variable.encrypted_predict_score.remote(en_scores + rand, suffix=(i, 'en_score'))
            masked_score = self.transfer_variable.masked_predict_score.get(idx=0, suffix=(i, 'masked_score'))
            final_score = masked_score - rand
            self.transfer_variable.final_predict_score.remote(final_score, suffix=(i, 'final_score'))

        return None

    def export_model(self):
        model_param = self.get_model_param()
        model_param.phi_a.extend(self.phi.tolist()[0])
        return {"FTLHostMeta": model_param, "FTLHostParam": self.get_model_param()}

    def load_model(self, model_dict):
        model_param = None
        model_meta = None
        for _, value in model_dict["model"].items():
            for model in value:
                if model.endswith("Meta"):
                    model_meta = value[model]
                if model.endswith("Param"):
                    model_param = value[model]
        LOGGER.info("load model")

        self.set_model_meta(model_meta)
        self.set_model_param(model_param)
        self.phi = np.array([model_param.phi_a])
