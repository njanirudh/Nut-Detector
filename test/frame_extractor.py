import cv2
import numpy as np

if __name__ == "__main__":
    VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_305.avi" #20
    # VIDEO = "/home/nj/HBRS/Studies/Sem-3/CV/Dataset/Videos/CV19_video_250.avi"

    cap = cv2.VideoCapture(VIDEO)

    old_hash = None
    old_image = None

    count = 0
    STUPID_FLAG = False

    while True:

        ret, frame = cap.read()
        if ret == True:

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(640,480))
            gray = cv2.blur(gray,(5,5))

            # cv2.imshow('frame', gray)
            count += 1

            # hsh = cv2.img_hash.PHash_create()
            hsh = cv2.img_hash.AverageHash_create()

            current_hash = hsh.compute(gray)

            if old_hash is not None :
                # print("Frame No:", count)
                # print(hsh.compare(current_hash,old_hash))
                diff_image = cv2.absdiff(old_image,gray)
                # print(np.sum(diff_image == 0)/(640*480))
                unique = len(np.unique(diff_image))
                print(unique)
                print()

                if unique > 150 :
                    print("--->", count)
                    STUPID_FLAG = True

                if STUPID_FLAG :
                    if unique < 15:
                        print("--->",count)
                        break

            old_hash = current_hash
            old_image = gray

            if cv2.waitKey(500) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()



