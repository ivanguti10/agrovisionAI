import { DiseaseInfo } from '../model/disease-info.model';

export const DISEASES_INFO: { [key: string]: DiseaseInfo } = {
    'Apple scab': {
        title: 'APPLE_SCAB_TITLE',
        imageUrl: '../../../assets/img/plagas/apple scab.jpg',
        description: 'APPLE_SCAB_DESCRIPTION_P',
        confidence: ''
    },
    'Apple Black Rot': {
      title: 'APPLE_BLACK_ROT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/apple black rot.jpg',
      description: 'APPLE_BLACK_ROT_DESCRIPTION_P',
      confidence: ''
  },
  'Manzano Roya del manzano y del cedro': {
      title: 'APPLE_CEDAR_RUST_TITLE_P',
      imageUrl: '../../../assets/img/plagas/roya cedro.jpg',
      description: 'APPLE_CEDAR_RUST_DESCRIPTION_P',
      confidence: ''
  },
  'Manzano saludable': {
      title: 'HEALTHY_APPLE_TITLE_P',
      imageUrl: '../../../assets/img/plagas/manzano sana.jpg',
      description: 'HEALTHY_APPLE_DESCRIPTION_P',
      confidence: ''
  },
  'Arándano saludable': {
      title: 'HEALTHY_BLUEBERRY_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Arandano.png',
      description: 'HEALTHY_BLUEBERRY_DESCRIPTION_P',
      confidence: ''
  },
  'Cereza (incluyendo ácida) Oidio': {
      title: 'CHERRY_POWDERY_MILDEW_TITLE_P',
      imageUrl: '../../../assets/img/plagas/oidio cerezo.jpg',
      description: 'CHERRY_POWDERY_MILDEW_DESCRIPTION_P',
      confidence: ''
  },
  'Cereza (incluyendo ácida) saludable': {
      title: 'HEALTHY_CHERRY_TITLE_P',
      imageUrl: '../../../assets/img/plagas/cereza.jpg',
      description: 'HEALTHY_CHERRY_DESCRIPTION_P',
      confidence: ''
  },
  'Vid Podredumbre negra': {
      title: 'GRAPE_BLACK_ROT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Prodedumbre negra de la Uva.jpg',
      description: 'GRAPE_BLACK_ROT_DESCRIPTION_P',
      confidence: ''
  },
  'Vid Esca (Sarampión negro)': {
      title: 'GRAPE_ESCA_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Madera de la vid.jpg',
      description: 'GRAPE_ESCA_DESCRIPTION_P',
      confidence: ''
  },
  'Vid Tizón de la hoja (Mancha de la hoja de Isariopsis)': {
      title: 'GRAPE_LEAF_BLIGHT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Tizon de la Vid.png',
      description: 'GRAPE_LEAF_BLIGHT_DESCRIPTION_P',
      confidence: ''
  },
  'Vid saludable': {
      title: 'HEALTHY_GRAPE_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Hoja de Vid sana.jpg',
      description: 'HEALTHY_GRAPE_DESCRIPTION_P',
      confidence: ''
  },
  'Naranja Huanglongbing (Greening de los cítricos)': {
      title: 'YELLOW_DRAGON_DISEASE_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Enfermedad del Dragon Amarillo.jpg',
      description: 'YELLOW_DRAGON_DISEASE_DESCRIPTION_P',
      confidence: ''
  },
  'Melocotón Mancha bacteriana': {
      title: 'PEACH_BACTERIAL_SPOT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Mancha bacteriana del Melocoton.jpg',
      description: 'PEACH_BACTERIAL_SPOT_DESCRIPTION_P',
      confidence: ''
  },
  'Melocotón saludable': {
      title: 'HEALTHY_PEACH_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Melocoton.jpeg',
      description: 'HEALTHY_PEACH_DESCRIPTION_P',
      confidence: ''
  },
  'Pimiento, morrón Mancha bacteriana': {
      title: 'PEPPER_BACTERIAL_SPOT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Mancha bacteriana Pimiento.jpg',
      description: 'PEPPER_BACTERIAL_SPOT_DESCRIPTION_P',
      confidence: ''
  },
  'Pimiento, morrón saludable': {
      title: 'HEALTHY_PEPPER_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Pimiento sano.jpg',
      description: 'HEALTHY_PEPPER_DESCRIPTION_P',
      confidence: ''
  },
  'Patata Tizón temprano': {
      title: 'POTATO_EARLY_BLIGHT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Tizon temprano de la Patata.png',
      description: 'POTATO_EARLY_BLIGHT_DESCRIPTION_P',
      confidence: ''
  },
  'Patata Tizón tardío': {
      title: 'POTATO_LATE_BLIGHT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Tizon tardio de Patata.jpeg',
      description: 'POTATO_LATE_BLIGHT_DESCRIPTION_P',
      confidence: ''
  },
  'Patata saludable': {
      title: 'HEALTHY_POTATO_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Patata.jpg',
      description: 'HEALTHY_POTATO_DESCRIPTION_P',
      confidence: ''
  },
  'Frambuesa saludable': {
      title: 'HEALTHY_RASPBERRY_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Frambuesa.jpg',
      description: 'HEALTHY_RASPBERRY_DESCRIPTION_P',
      confidence: ''
  },
  'Soja saludable': {
      title: 'HEALTHY_SOYBEAN_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Frijol.jpeg',
      description: 'HEALTHY_SOYBEAN_DESCRIPTION_P',
      confidence: ''
  },
  'Calabaza Oidio': {
      title: 'SQUASH_POWDERY_MILDEW_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Oidio de Calabaza.jpg',
      description: 'SQUASH_POWDERY_MILDEW_DESCRIPTION_P',
      confidence: ''
  },
  'Fresa Chamusco de la hoja': {
      title: 'STRAWBERRY_LEAF_SCORCH_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Quemadura de Fresa.jpg',
      description: 'STRAWBERRY_LEAF_SCORCH_DESCRIPTION_P',
      confidence: ''
  },
  'Fresa saludable': {
      title: 'HEALTHY_STRAWBERRY_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Fresa.jpg',
      description: 'HEALTHY_STRAWBERRY_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Mancha bacteriana': {
      title: 'TOMATO_BACTERIAL_SPOT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Mancha bacteriana del Tomate.jpg',
      description: 'TOMATO_BACTERIAL_SPOT_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Tizón temprano': {
      title: 'TOMATO_EARLY_BLIGHT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Tizon temprano del Tomate.jpeg',
      description: 'TOMATO_EARLY_BLIGHT_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Tizón tardío': {
      title: 'TOMATO_LATE_BLIGHT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Tizon tardio del Tomate.jpg',
      description: 'TOMATO_LATE_BLIGHT_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Moho de la hoja': {
      title: 'TOMATO_LEAF_MOLD_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Mildiu del Tomate.jpg',
      description: 'TOMATO_LEAF_MOLD_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Mancha foliar de Septoria': {
      title: 'TOMATO_SEPTORIA_LEAF_SPOT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Septoriosis del Tomate.jpeg',
      description: 'TOMATO_SEPTORIA_LEAF_SPOT_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Araña roja (Ácaro de dos manchas)': {
      title: 'TOMATO_TWO_SPOTTED_SPIDER_MITE_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Araña roja del Tomate.jpg',
      description: 'TOMATO_TWO_SPOTTED_SPIDER_MITE_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Mancha diana': {
      title: 'TOMATO_TARGET_SPOT_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Mancha de diana del Tomate.jpg',
      description: 'TOMATO_TARGET_SPOT_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Virus del rizado amarillo de la hoja del tomate': {
      title: 'TOMATO_YELLOW_LEAF_CURL_VIRUS_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Virus del rizado amarillo del tomate.jpeg',
      description: 'TOMATO_YELLOW_LEAF_CURL_VIRUS_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate Virus del mosaico del tomate': {
      title: 'TOMATO_MOSAIC_VIRUS_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Virus del mosaico del tomate.jpg',
      description: 'TOMATO_MOSAIC_VIRUS_DESCRIPTION_P',
      confidence: ''
  },
  'Tomate saludable': {
      title: 'HEALTHY_TOMATO_TITLE_P',
      imageUrl: '../../../assets/img/plagas/Tomate.jpg',
      description: 'HEALTHY_TOMATO_DESCRIPTION_P',
      confidence: ''
  }
  
}
