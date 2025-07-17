import pandas as pd
import glob
import plot_trajectories as pt

def get_data_and_matrix(dfpath='C:/Users/Alexandre/Desktop/CRM/Alex/paper/',
                        after_correct=True, silent=False, return_df=False, srfail=False):
    """
    Import data for 1 rat.
    """
    print('Loading data')
    files = glob.glob(dfpath+'*.pkl')
    for f in files:
        df = pd.read_pickle(f)
        if return_df:
            if after_correct:
                if not silent:
                    if srfail:
                        return df.query(
                               "sound_len <= 400 and\
                                   resp_len <=1 and R_response>= 0\
                                       and hithistory >= 0 and special_trial == 0\
                                           and aftererror==0")
                    else:
                        return df.query(
                                "sound_len <= 400 and soundrfail ==\
                                    False and resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0 and special_trial == 0\
                                            and aftererror==0")
                if silent:
                    if srfail:
                        return df.query(
                                "sound_len <= 400 and \
                                    resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0\
                                            and aftererror==0")
                    else:
                        return df.query(
                                "sound_len <= 400 and soundrfail ==\
                                    False and resp_len <=1 and R_response>= 0\
                                        and hithistory >= 0\
                                            and aftererror==0")
            else:
                if not silent:
                    if srfail:
                        return df.query(
                            "sound_len <= 400 and\
                            resp_len <=1 and R_response>= 0\
                            and hithistory >= 0 and special_trial == 0")
                    else:
                        return df.query(
                            "sound_len <= 400 and soundrfail ==\
                                False and resp_len <=1 and R_response>= 0\
                                    and hithistory >= 0 and special_trial == 0")
                if silent:
                    if srfail:
                        return df.query(
                         "sound_len <= 400 and\
                         resp_len <=1 and R_response>= 0\
                         and hithistory >= 0")
                    else:
                        return df.query(
                            "sound_len <= 400 and soundrfail ==\
                            False and resp_len <=1 and R_response>= 0\
                            and hithistory >= 0")
  
def process_data():
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)  # prior
    df['allpriors'] = zt
    hit = np.array(df['hithistory'])  #hit, 1 if correct, 0 if incorrect
    stim = np.array([stim for stim in df.res_sound])  # stim, in frames of 50 ms, structure: N trials x 20 frames
    coh = np.array(df.coh2)  # putative coherence [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]
    decision = np.array(df.R_response) * 2 - 1  # decision, 1 if right, 0 if left
    traj_stamps = df.trajectory_stamps.values  # time index of the trajectory points
    traj_y = df.trajectory_y.values  # trajectory y(t)
    fix_onset = df.fix_onset_dt.values  # value in time of the fixation onset
    fix_breaks = np.vstack(np.concatenate([df.sound_len/1000,
                                            np.concatenate(df.fb.values)-0.3]))  #all reaction times including fixation breaks (RT<0)
    sound_len = np.array(df.sound_len)  # reaction time (RT) without fixation breaks
    gt = np.array(df.rewside) * 2 - 1  # ground truth, 1 if right, -1 if left
    trial_index = np.array(df.origidx)  # integer from 1 to trial at end of session
    resp_len = np.array(df.resp_len)  # movement time (MT)
    time_trajs = edd2.get_trajs_time(resp_len=resp_len,
                                    traj_stamps=traj_stamps,
                                    fix_onset=fix_onset, com=None,
                                    sound_len=sound_len)  # to extract the timing of each timepoint of the trajectories
    special_trial = df.special_trial  # 0 if normal, 2 if silent
    df['time_trajs'] = time_trajs
    subjid = df.subjid.values  # subject name
    print('Computing CoMs')
    time_com, peak_com, com =\
        fig_3.com_detection(df=df, data_folder=DATA_FOLDER, save_dat=True,
                            com_threshold=com_threshold)  # compute CoMs given y(t) and a threshold
    print('Ended Computing CoMs')
    com = np.array(com)  # new CoM list, True if CoM, False if non-CoM
    df['CoM_sugg'] = com
    df['norm_allpriors'] = fp.norm_allpriors_per_subj(df)  # normalize prior by max value for each subject
    df['time_trajs'] = time_trajs
                      
if __name__ == "__main__":
    # Example usage
    path = '/home/manuel/Descargas/LE41'
    df = get_data_and_matrix(dfpath=path,
                             return_df=True, after_correct=False, silent=False)
    
    # df1 and df2 are your two DataFrames
    relevant_cols = ['origidx', 'rewside', 'hithistory', 'R_response', 'subjid', 'sessid',
       'resp_len', 'res_sound', 'sound_len', 'aftererror', 'fb', 'soundrfail',
       'special_trial', 'fix_onset_dt', 'trajectory_y', 'trajectory_x',
       'trajectory_stamps', 'CoM_sugg', 'framerate', 'dW_lat', 'dW_trans',
       'traj_d1', 'avtrapz', 'coh2', 'time_trajs']
    print(f"Common columns: {relevant_cols}")
    # keep only the shared columns in df1
    df = df[[c for c in relevant_cols if c in df.columns]]
    print(df.head())

    pt.fig_2_trajs(df=df.loc[df.soundrfail == 0])