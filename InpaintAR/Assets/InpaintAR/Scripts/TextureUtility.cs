using UnityEngine;

namespace InpaintAR.Scripts {
    public static class TextureUtility {
        // Size should always stay the same during program, only mask changes; -> Initialize empty once and then reuse to avoid reinitialization each frame
        private static int _mImageWidth = -1;
        private static int _mImageHeight = -1;

        private static Color[] _mEmptyImage;

        public static Color[] GetEmptyImagePixels(Texture fillImageTexture) {
            if (_mEmptyImage is null) {
                InitializeEmptyPixels(fillImageTexture);
            }

            return _mEmptyImage;
        }

        public static int GetImageWidth(Texture fillImageTexture) {
            if (_mEmptyImage is null) {
                InitializeEmptyPixels(fillImageTexture);
            }

            return _mImageWidth;
        }
        
        public static int GetImageHeight(Texture fillImageTexture) {
            if (_mEmptyImage is null) {
                InitializeEmptyPixels(fillImageTexture);
            }

            return _mImageHeight;
        }
        
        private static void InitializeEmptyPixels(Texture fillImageTexture) {
            _mImageWidth  = fillImageTexture.width;
            _mImageHeight = fillImageTexture.height;
            _mEmptyImage  = new Color[_mImageWidth * _mImageHeight];

            for (int i = 0; i < _mEmptyImage.Length; i++) {
                _mEmptyImage[i] = Color.red;
            }
        }
    }
}