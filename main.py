from utils import *

def main(batch_size=1, saturate_threshold=1.6, max_iteration=100, stop_encode_threshold=3.5):
    result_dir = get_time()
    os.makedirs(result_dir, exist_ok=True)
    data_dir = './img'
    sess, image, loss, optimizer, var_list = build(batch_size, saturate_threshold)

    k = 1
    for root, _, files in os.walk(data_dir):
        for file in files:
            image_values = np.array(PIL.Image.open(root + '/' + file).resize((1024,1024),PIL.Image.ANTIALIAS)).astype(np.float32)[np.newaxis, ...]       

            sess.run(tf.variables_initializer(var_list))
            print('\n----Encoding----')
            loss_encoder, i = 100, 0
            while loss_encoder >= stop_encode_threshold and i <= 500:
                loss_encoder, _ = sess.run([loss['image'], optimizer['encoder']], {image['real']: image_values})
                loss_encoder = np.mean(loss_encoder)
                print('[ Encoding %d ]\t[ Loss %.4g ]' % (i, loss_encoder), end='\r')
                i += 1

            print('\n----Training----')
            log_file = open('log{}.txt'.format(k), 'w')
            for epoch in range(max_iteration):
                sess.run((optimizer['delta'], optimizer['weight']), {image['real']: image_values})
                loss_val = sess.run(loss, {image['real']: image_values})
                output({'Epoch': str(epoch+1) + '/' + str(max_iteration), 'Loss_i': np.mean(loss_val['image']), 'Dist_n': loss_val['face_sat']}, log_file, bit=6)   

            adversarial_samples = sess.run(image['adv255'])
            for i in range(batch_size):
                PIL.Image.fromarray(adversarial_samples[i].astype(np.uint8)).save(result_dir + '/adv{}.png'.format(k))
                per = np.clip((adversarial_samples[i].astype(np.float32) - image_values[i].astype(np.float32))*3, 0, 255)
                PIL.Image.fromarray(per.astype(np.uint8)).save(result_dir + '/per_times_3_{}.png'.format(k))

            print('\n----Done!----')
            log_file.close()
            k = k + 1

    sess.close()
    tf.reset_default_graph()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    main()
