import pandas as pd
import glob
import plot_trajectories as pt
import numpy as np
import coms
import auxiliary_functions as af

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

def get_trajs_time(traj_stamps, fix_onset, com, sound_len,
                   com_cond=False):
    time = []
    if com_cond:
        traj_st_com = traj_stamps[com.astype(bool)]
        fix_onset_com = fix_onset[com.astype(bool)]
        sound_len_com = sound_len[com.astype(bool)]
    else:
        traj_st_com = traj_stamps
        fix_onset_com = fix_onset
        sound_len_com = sound_len
    for j in range(len(traj_st_com)):
        t = traj_st_com[j] - fix_onset_com[j]
        t = (t.astype(int) / 1000_000 - 300 - sound_len_com[j])
        time.append(t)
    return np.array(time, dtype='object')


def process_data(data_folder, com_threshold=8):
    zt = np.nansum(df[["dW_lat", "dW_trans"]].values, axis=1)  # prior
    df['allpriors'] = zt
    traj_stamps = df.trajectory_stamps.values  # time index of the trajectory points
    fix_onset = df.fix_onset_dt.values  # value in time of the fixation onset
    sound_len = np.array(df.sound_len)  # reaction time (RT) without fixation breaks
    time_trajs = get_trajs_time(traj_stamps=traj_stamps,
                                fix_onset=fix_onset, com=None,
                                sound_len=sound_len)  # to extract the timing of each timepoint of the trajectories
    df['time_trajs'] = time_trajs
    print('Computing CoMs')
    _, _, com =\
        coms.com_detection(df=df, data_folder=data_folder, save_dat=True,
                            com_threshold=com_threshold)  # compute CoMs given y(t) and a threshold
    print('Ended Computing CoMs')
    com = np.array(com)  # new CoM list, True if CoM, False if non-CoM
    df['CoM_sugg'] = com
    df['norm_allpriors'] = af.norm_allpriors_per_subj(df)  # normalize prior by max value for each subject
    df['time_trajs'] = time_trajs
    return df
                      
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
    df = process_data(data_folder=path, com_threshold=8)
    pt.fig_2_trajs(df=df.loc[df.soundrfail == 0])