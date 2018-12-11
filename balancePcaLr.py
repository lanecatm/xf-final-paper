
    from imblearn.combine import SMOTETomek
    from imblearn.under_sampling import NearMiss
    # smote_tomek = SMOTETomek(random_state=0)
    nm = NearMiss()
    print('resampling...')
    X_resampled, y_resampled = nm.fit_resample(train_X, train_y)
