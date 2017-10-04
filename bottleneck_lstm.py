





def get_lstm_frame(vid, i_frame, step=1, fr_width=3):
    vid_length = vid.get_length()
    rel_ind_range = range(-(fr_width // 2) * step, (fr_width // 2) * step + 1, step)

    frame_0 = vid.get_data(0)
    frame_arr = np.zeros((fr_width, *frame_0.shape))
    for i, rel_ind in enumerate(rel_ind_range):
        i_fr = i_frame + rel_ind
        # Achtung sometimes it's not true
        if i_fr < 0 or i_fr >= vid_length:
            i_fr = i_frame + (-1) * rel_ind

        frame_arr[i] = vid.get_data(i_fr)

    return frame_arr


def get_best_video_id_hash(df, img_hash, step=1, fr_width=3):
    if fr_width % 2 == 0:
        print('Achtung, must be fr_width % 2 != 0')
        return

    df_hash_cur = df[df['hash'] == img_hash].copy()

    rel_ind_range = range(-(fr_width // 2) * step, (fr_width // 2) * step + 1, step)
    best_vid_score = 0
    best_video_id = None
    best_i_frame = 0
    for i_row, row in df_hash_cur.iterrows():

        video_path = os.path.join(train_dir, row.video_id + '.mp4')
        vid = imageio.get_reader(video_path, 'ffmpeg')
        vid_length = vid.get_length()

        fr_indexes = [rel_ind + row.frame for rel_ind in rel_ind_range if
                      rel_ind + row.frame > 0 and rel_ind + row.frame < vid_length]

        if best_vid_score < len(fr_indexes) or best_video_id is None:
            best_vid_score = len(fr_indexes)
            best_video_id = row.video_id
            best_i_frame = row.frame

        if best_vid_score == fr_width:
            break

    return best_video_id, best_i_frame


def get_bad_frames(df, vid, n_bad_frames=2):
    i_frames = list(range(vid.get_length()))
    for i_fr in df[df.video_id == video_id].frame:
        i_frames.remove(i_fr)

    frame_arr = []
    for _ in range(1000):
        i_fr = np.random.choice(i_frames, 1)[0]
        frame_arr.append(i_fr)
        i_frames.remove(i_fr)

        if len(frame_arr) == n_bad_frames:
            break

    return frame_arr


def save_lstm_frame(lstm_frame, path, i_cl):
    n_sub_fr = lstm_frame.shape[0]

    frame_list_crop = []
    for i in range(n_sub_fr):
        frame_crop = utils.crop_ruler_area_short(lstm_frame[i], i_cl)
        frame_list_crop.append(frame_crop)

    frame_stack_crop = np.zeros((n_sub_fr, *frame_list_crop[0].shape))

    for i, fr in enumerate(frame_list_crop):
        frame_stack_crop[i] = fr

    np.save(path, frame_stack_crop.astype('float32'))